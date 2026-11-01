"""Typed Kalshi V2 REST client for a configurable exchange, primary subaccount 0.

Protocol references:
https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
https://docs.kalshi.com/api-reference/orders/create-order-v2
https://docs.kalshi.com/api-reference/portfolio/get-balance

RSA-PSS/SHA-256 or Ed25519 signs timestamp_ms + METHOD + full API path,
excluding the query. Prices/quantities on the wire are fixed-point strings.
Only GETs are retried. An ambiguous POST is NEVER automatically repeated.
The client defaults to paper mode; construction performs no network requests.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from email.utils import parsedate_to_datetime
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Literal, TypeAlias
from urllib.parse import urlsplit
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from cryptography.exceptions import UnsupportedAlgorithm
import requests

from kalshi_ml.config import KalshiEnvironment, RuntimeConfig, Settings, TradingMode

API_ROOT = "/trade-api/v2"
ORDER_PATH = f"{API_ROOT}/portfolio/events/orders"
BASE_URLS = {
    KalshiEnvironment.PROD: "https://external-api.kalshi.com",
    KalshiEnvironment.DEMO: "https://external-api.demo.kalshi.co",
}
SigningKey: TypeAlias = rsa.RSAPrivateKey | ed25519.Ed25519PrivateKey
Numeric: TypeAlias = str | int | float | Decimal
TargetSide: TypeAlias = Literal["YES", "NO"]


class KalshiError(RuntimeError):
    """Base error; messages intentionally omit headers, key data and API bodies."""


class AuthenticationError(KalshiError):
    """Credentials are missing, invalid, or rejected by the exchange."""


class InvalidResponseError(KalshiError):
    """The exchange response cannot be interpreted without guessing."""


class TradingDisabledError(KalshiError):
    """An order was attempted while the configured execution mode disallows it."""


class APIError(KalshiError):
    """An HTTP error with sanitized status and optional order correlation ID."""

    def __init__(self, status_code: int, *, client_order_id: str | None = None) -> None:
        self.status_code = status_code
        self.client_order_id = client_order_id
        super().__init__(f"Kalshi HTTP {status_code}")


class TransportError(KalshiError):
    """A GET could not complete within its transport/retry budget."""


class RateLimitError(APIError):
    """Throttled request; retry_after_sec is advisory, never an automatic POST retry."""

    def __init__(self, retry_after_sec: float, *, client_order_id: str | None = None) -> None:
        super().__init__(429, client_order_id=client_order_id)
        self.retry_after_sec = retry_after_sec


class OrderSubmissionUnknown(KalshiError):
    """The order may exist: reconcile this ID before considering another order.

    This includes transport failures, 408/5xx and malformed success responses.
    A new send_order call normally creates a NEW ID and could duplicate exposure.
    """

    def __init__(self, client_order_id: str) -> None:
        self.client_order_id = client_order_id
        super().__init__(f"Order outcome unknown; reconcile client_order_id={client_order_id}")


def _decimal(value: object, field: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    """Parse a finite numeric value without accepting booleans or nulls."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InvalidResponseError(f"invalid {field}")
    if isinstance(value, str) and len(value) > 128:
        raise InvalidResponseError(f"oversized {field}")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise InvalidResponseError(f"invalid {field}") from None
    if not result.is_finite() or result < minimum or result > Decimal("1e24"):
        raise InvalidResponseError(f"nonfinite or out-of-range {field}")
    return result


def safe_to_cents(
    value: Numeric,
    *,
    unit: Literal["dollars", "cents"] = "dollars",
    rounding: Literal["exact", "floor", "ceil"] = "exact",
) -> int:
    """Parse a binary-contract quote into cents with explicit units.

    ``safe_to_cents("0.5600") == 56``; legacy ``"56"`` needs unit="cents".
    No value-magnitude heuristic is used ("1" could mean $1 or one cent).
    Default rejects fractional cents. Quote bids may explicitly round down,
    asks up, preserving conservative executable prices on sub-cent markets.
    Missing/invalid prices raise; they never become a tradable zero by default.
    """
    if unit not in {"dollars", "cents"} or rounding not in {"exact", "floor", "ceil"}:
        raise ValueError("invalid price unit or rounding mode")
    number = _decimal(value, "quote price")
    cents = number * 100 if unit == "dollars" else number
    if cents > 100:
        raise InvalidResponseError("quote price outside [0, 100] cents")
    if rounding == "exact" and cents != cents.to_integral_value():
        raise InvalidResponseError("quote price contains fractional cents")
    rounded = cents.to_integral_value(rounding=ROUND_CEILING if rounding == "ceil" else ROUND_FLOOR)
    return int(rounded)


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("require a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _date(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidResponseError(f"missing or invalid {field}")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        raise InvalidResponseError(f"invalid {field}") from None


def _index(value: object, field: str) -> int:
    number = _decimal(value, field)
    if number != number.to_integral_value():
        raise InvalidResponseError(f"nonintegral {field}")
    return int(number)


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", value):
        raise ValueError(f"invalid {field}")
    return value


def load_signing_key(path: str | Path, *, password: bytes | None = None) -> SigningKey:
    """Load a PEM private key; reject unsupported algorithms and small RSA keys."""
    try:
        key = serialization.load_pem_private_key(Path(path).expanduser().read_bytes(), password=password)
    except (ValueError, TypeError, OSError, UnsupportedAlgorithm):
        raise AuthenticationError("cannot load PEM signing key") from None
    return _validate_key(key)


def _validate_key(key: object) -> SigningKey:
    if not isinstance(key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
        raise AuthenticationError("only RSA and Ed25519 private keys are supported")
    if isinstance(key, rsa.RSAPrivateKey) and key.key_size < 2048:
        raise AuthenticationError("RSA key must be at least 2048 bits")
    return key


@dataclass(frozen=True, slots=True)
class ClientPolicy:
    """Bounded GET retry and pacing policy; POSTs always have one attempt."""

    timeout_sec: float = 10.0
    max_read_retries: int = 3
    base_delay_sec: float = 0.5
    max_delay_sec: float = 30.0
    min_request_interval_sec: float = 0.1
    max_market_pages: int = 100

    def __post_init__(self) -> None:
        for name in ("timeout_sec", "base_delay_sec", "max_delay_sec", "min_request_interval_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "min_request_interval_sec" and value == 0):
                raise ValueError(f"invalid {name}")
        if self.base_delay_sec > self.max_delay_sec:
            raise ValueError("base delay exceeds max delay")
        if type(self.max_read_retries) is not int or not 0 <= self.max_read_retries <= 10:
            raise ValueError("max_read_retries must be an integer in [0, 10]")
        if type(self.max_market_pages) is not int or not 1 <= self.max_market_pages <= 1000:
            raise ValueError("max_market_pages must be an integer in [1, 1000]")


@dataclass(frozen=True, slots=True)
class Balance:
    """Available cash on the configured shard, in exact dollars; holdings are separate."""

    balance_dollars: Decimal
    portfolio_value_dollars: Decimal | None
    updated_at: datetime | None
    exchange_index: int = 2
    subaccount: int = 0

    @property
    def balance_cents(self) -> Decimal:
        """Retain sub-cent balance precision; do not round spending capacity up."""
        return self.balance_dollars * 100


@dataclass(frozen=True, slots=True)
class ActiveMarket:
    """Currently observable cycle on the configured shard; missing quotes/strike stay None.

    open_time is the listing's trading-open time, which may precede the actual
    15-minute observation cycle. cycle_start_ts is close_time minus 15 minutes.
    Quotes are conservatively rounded to whole cents; strike is a USD float.
    Missing strike must prevent a later strategy from opening a position.
    """

    ticker: str
    event_ticker: str
    status: str
    open_time: datetime
    close_time: datetime
    cycle_start_ts: datetime
    observed_at: datetime
    yes_bid_cents: int | None
    yes_ask_cents: int | None
    no_bid_cents: int | None
    no_ask_cents: int | None
    strike_price: float | None
    strike_type: str | None
    volume: Decimal | None
    exchange_index: int = 2


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Matching-engine acknowledgement; acknowledgement is not a full fill."""

    order_id: str
    client_order_id: str
    fill_count: Decimal
    remaining_count: Decimal
    processed_at: datetime
    average_yes_fill_price_dollars: Decimal | None
    average_fee_paid_dollars: Decimal | None


@dataclass(frozen=True, slots=True)
class OrderBook:
    """Best executable prices and displayed quantities, without invented depth."""

    ticker: str
    observed_at: datetime
    yes_bid: Decimal | None
    yes_bid_size: Decimal
    no_bid: Decimal | None
    no_bid_size: Decimal

    def quote(self, side: TargetSide, *, closing: bool) -> tuple[Decimal | None, Decimal]:
        """Return the held leg's bid for exits, implied ask for entries."""
        price, size = (self.yes_bid, self.yes_bid_size) if (side == "YES") == closing else (self.no_bid, self.no_bid_size)
        return (price if closing or price is None else Decimal(1) - price), size


@dataclass(frozen=True, slots=True)
class MarketLifecycle:
    """Official final payout stays unavailable until settled/finalized status."""

    ticker: str
    status: str
    close_time: datetime
    yes_payout: Decimal | None
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class FillTotals:
    """Exact aggregate of deduplicated exchange fills, in the requested leg."""

    quantity: Decimal
    notional: Decimal
    fees: Decimal


class KalshiClient:
    """Synchronous, reusable client; use one instance per worker/thread.

    Omit credentials for public market discovery. Provide both for account
    reads/orders. Injected sessions remain caller-owned and must not enable
    HTTP-adapter retries. Redirects are refused to keep signed credentials on
    the selected allowlisted host. Never log prepared requests/headers.
    """

    def __init__(
        self,
        *,
        environment: KalshiEnvironment = KalshiEnvironment.DEMO,
        key_id: str | None = None,
        private_key: SigningKey | None = None,
        trading_mode: TradingMode = TradingMode.PAPER,
        live_trading_enabled: bool = False,
        exchange_index: int = 2,
        policy: ClientPolicy | None = None,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        if not isinstance(environment, KalshiEnvironment):
            raise TypeError("environment must be a KalshiEnvironment")
        runtime = RuntimeConfig(mode=trading_mode, live_trading_enabled=live_trading_enabled, exchange_index=exchange_index)
        if trading_mode is TradingMode.DEMO and environment is not KalshiEnvironment.DEMO:
            raise ValueError("demo trading requires the demo host")
        if trading_mode is TradingMode.LIVE and environment is not KalshiEnvironment.PROD:
            raise ValueError("live trading requires the production host")
        if (key_id is None) != (private_key is None):
            raise AuthenticationError("provide both key ID and private key")
        if private_key is not None:
            _validate_key(private_key)
            assert key_id is not None
            _identifier(key_id, "key ID")
        self.environment, self.runtime = environment, runtime
        self._key_id, self._private_key = key_id, private_key
        self.policy = policy if policy is not None else ClientPolicy()
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        if self._owns_session:
            self._session.trust_env = False
        self._check_adapter_retries()
        self._clock, self._sleep, self._monotonic = clock, sleep, monotonic
        self._jitter, self._uuid_factory = jitter, uuid_factory
        self._last_request: float | None = None
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings, *, password: bytes | None = None, policy: ClientPolicy | None = None, **kwargs: Any) -> KalshiClient:
        """Bind the existing .env configuration; parse the key without network I/O."""
        settings.validate_credentials()
        key_path = settings.private_key_path
        if key_path is None:
            raise AuthenticationError("private key path is required")
        key = load_signing_key(key_path, password=password)
        return cls(
            environment=settings.environment, key_id=settings.key_id, private_key=key,
            trading_mode=settings.runtime.mode, live_trading_enabled=settings.runtime.live_trading_enabled,
            exchange_index=settings.runtime.exchange_index,
            policy=policy if policy is not None else ClientPolicy(timeout_sec=float(settings.http_timeout_sec), max_read_retries=settings.http_max_retries),
            **kwargs,
        )

    def __enter__(self) -> KalshiClient:
        if self._closed:
            raise KalshiError("client is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close only client-owned sessions; the client cannot be reused afterward."""
        if not self._closed and self._owns_session:
            self._session.close()
        self._closed = True

    def _check_adapter_retries(self) -> None:
        if self._session.verify is False:
            raise ValueError("TLS certificate verification must be enabled")
        for adapter in self._session.adapters.values():
            if adapter.max_retries.total not in (0, False):
                raise ValueError("session adapter retries must be disabled, including POST retries")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self._private_key is None or self._key_id is None:
            raise AuthenticationError("authenticated endpoint requires credentials")
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith(f"{API_ROOT}/") or parsed.fragment:
            raise ValueError("sign only an API-root-relative path")
        timestamp = str(int(_utc(self._clock()).timestamp() * 1000))
        message = f"{timestamp}{method.upper()}{parsed.path}".encode("utf-8")
        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            signature = self._private_key.sign(message)
        else:
            signature = self._private_key.sign(
                message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _retry_after(self, header: str | None) -> float:
        if header is None:
            return 0.0
        try:
            seconds = float(header)
        except ValueError:
            try:
                seconds = (_utc(parsedate_to_datetime(header)) - _utc(self._clock())).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return 0.0
        return max(0.0, seconds) if math.isfinite(seconds) else math.inf

    def _request(
        self, method: Literal["GET", "POST", "DELETE"], path: str, *, authenticated: bool = True,
        params: Mapping[str, str | int] | None = None, payload: dict[str, object] | None = None,
        client_order_id: str | None = None,
        expected_status: int | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise KalshiError("client is closed")
        if method != "GET" and client_order_id is None:
            raise ValueError("mutating requests require a client order ID")
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or not path.startswith(f"{API_ROOT}/"):
            raise ValueError("request path must be relative; pass query values through params")
        self._check_adapter_retries()
        attempts = 1 + self.policy.max_read_retries if method == "GET" else 1
        for attempt in range(attempts):
            wait = 0.0
            if self._last_request is not None:
                self._sleep(max(0.0, self.policy.min_request_interval_sec - (self._monotonic() - self._last_request)))
            headers = {"Accept": "application/json", "User-Agent": "kalshi-ml/0.1"}
            if authenticated:
                headers.update(self._auth_headers(method, path))
            if payload is not None:
                headers["Content-Type"] = "application/json"
            self._last_request = self._monotonic()
            try:
                with self._session.request(
                    method, BASE_URLS[self.environment] + path,
                    params=params, json=payload, headers=headers,
                    timeout=(min(5.0, self.policy.timeout_sec), self.policy.timeout_sec),
                    allow_redirects=False,
                ) as response:
                    status = response.status_code
                    wait = self._retry_after(response.headers.get("Retry-After"))
                    if status in (401, 403):
                        raise AuthenticationError(f"Kalshi authentication/permission rejected (HTTP {status})")
                    expected = expected_status if expected_status is not None else (201 if method == "POST" else 200)
                    if method != "GET" and (status in (408, 409) or status >= 500 or 300 <= status < 400 or 200 <= status < 300 and status != expected):
                        raise OrderSubmissionUnknown(client_order_id)
                    if status == 429:
                        if method != "GET" or attempt + 1 == attempts or wait > self.policy.max_delay_sec:
                            raise RateLimitError(wait, client_order_id=client_order_id)
                    elif method == "GET" and (status == 408 or 500 <= status < 600):
                        if attempt + 1 == attempts:
                            raise APIError(status)
                    elif 200 <= status < 300:
                        try:
                            body = response.json()
                        except ValueError:
                            if method != "GET":
                                raise OrderSubmissionUnknown(client_order_id) from None
                            raise InvalidResponseError("GET response is not JSON") from None
                        if not isinstance(body, dict):
                            if method != "GET":
                                raise OrderSubmissionUnknown(client_order_id)
                            raise InvalidResponseError("expected a JSON object")
                        return body
                    else:
                        raise APIError(status, client_order_id=client_order_id)
            except requests.RequestException:
                if method != "GET":
                    raise OrderSubmissionUnknown(client_order_id) from None
                if attempt + 1 == attempts:
                    raise TransportError("GET transport retry budget exhausted") from None
            if wait > self.policy.max_delay_sec:
                raise TransportError("GET Retry-After exceeds the configured wait budget")
            delay = min(self.policy.max_delay_sec, self.policy.base_delay_sec * 2**attempt * (1 + 0.25 * self._jitter()))
            self._sleep(max(wait, delay))
        raise AssertionError("unreachable request state")

    def get_balance(self, *, exchange_index: int | None = None) -> Balance:
        """Return the configured shard balance, never an aggregate from another shard.

        Breakdown.balance is a dollar string, legacy top-level balance is cents.
        If breakdown is absent, the explicitly exchange-scoped top-level balance
        is used (restricted keys may omit breakdown). Present but missing or
        duplicate configured-shard entries raise instead of falling back to an aggregate.
        """
        index = self.runtime.exchange_index if exchange_index is None else RuntimeConfig(exchange_index=exchange_index).exchange_index
        body = self._request("GET", f"{API_ROOT}/portfolio/balance", params={"subaccount": 0, "exchange_index": index})
        if "balance_breakdown" in body:
            breakdown = body["balance_breakdown"]
            if not isinstance(breakdown, list):
                raise InvalidResponseError("invalid balance breakdown")
            matches: list[dict[str, Any]] = []
            for row in breakdown:
                if not isinstance(row, dict):
                    raise InvalidResponseError("invalid balance breakdown entry")
                if _index(row.get("exchange_index"), "exchange index") == index:
                    if "subaccount" not in row or _index(row["subaccount"], "subaccount") == 0:
                        matches.append(row)
            if len(matches) != 1:
                raise InvalidResponseError("expected exactly one balance entry for the configured shard")
            balance = _decimal(matches[0].get("balance"), "shard balance")
        elif "balance_dollars" in body:
            balance = _decimal(body["balance_dollars"], "balance dollars")
        else:
            balance = _decimal(body.get("balance"), "legacy balance cents") / 100
        portfolio = _decimal(body["portfolio_value"], "portfolio value cents") / 100 if body.get("portfolio_value") is not None else None
        updated = None
        if body.get("updated_ts") is not None:
            try:
                updated = datetime.fromtimestamp(_index(body["updated_ts"], "balance timestamp"), timezone.utc)
            except (ValueError, OverflowError, OSError):
                raise InvalidResponseError("invalid balance timestamp") from None
        return Balance(balance, portfolio, updated, exchange_index=index)

    def transfer_collateral(self, amount_dollars: Decimal, *, client_transfer_id: str) -> str:
        """One asynchronous primary-account transfer, event-contract shard 0 -> 2.

        Amount is converted exactly to centicents (USD * 10,000). An ambiguous
        response carries the client transfer ID through OrderSubmissionUnknown;
        callers must reconcile transfer history before any further attempt.
        """
        if self.runtime.mode is not TradingMode.LIVE or not self.runtime.live_trading_enabled or self.environment is not KalshiEnvironment.PROD:
            raise TradingDisabledError("collateral transfers require explicitly enabled production live mode")
        if not isinstance(amount_dollars, Decimal) or not amount_dollars.is_finite() or amount_dollars <= 0:
            raise ValueError("amount must be a positive finite Decimal")
        centicents = amount_dollars * 10_000
        if centicents != centicents.to_integral_value() or centicents > 2**63 - 1:
            raise ValueError("amount must fit int64 centicents with at most four decimal places")
        transfer_id = str(uuid.UUID(client_transfer_id))
        payload: dict[str, object] = {
            "source": "event_contract", "destination": "event_contract", "amount": int(centicents),
            "source_exchange_shard": 0, "destination_exchange_shard": 2,
            "source_subaccount": 0, "destination_subaccount": 0, "client_transfer_id": transfer_id,
        }
        body = self._request("POST", f"{API_ROOT}/portfolio/intra_exchange_instance_transfer", payload=payload,
                             client_order_id=transfer_id, expected_status=200)
        server_id = body.get("transfer_id")
        if not isinstance(server_id, str) or not server_id:
            raise OrderSubmissionUnknown(transfer_id)
        return server_id

    @staticmethod
    def _quote(row: Mapping[str, Any], name: str, *, ask: bool) -> int | None:
        rounding: Literal["floor", "ceil"] = "ceil" if ask else "floor"
        if row.get(f"{name}_dollars") is not None:
            return safe_to_cents(row[f"{name}_dollars"], rounding=rounding)
        if row.get(name) is not None:
            return safe_to_cents(row[name], unit="cents", rounding=rounding)
        return None

    def get_active_15m_market(self, series_ticker: str = "KXBTC15M") -> ActiveMarket | None:
        """Paginate and select the earliest-closing currently active UTC cycle.

        The API's `status=open` query can yield status="active" objects. Both
        representations are accepted, but future listing/cycle times, expired,
        paused, resolved, provisional and other-shard markets are rejected.
        No future cycle is returned merely because it has already been listed.
        Missing numeric fields remain None; malformed numbers raise.
        """
        _identifier(series_ticker, "series ticker")
        cursor = ""
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _ in range(self.policy.max_market_pages):
            params: dict[str, str | int] = {"series_ticker": series_ticker, "status": "open", "limit": 1000, "mve_filter": "exclude"}
            if cursor:
                params["cursor"] = cursor
            body = self._request("GET", f"{API_ROOT}/markets", params=params, authenticated=False)
            markets = body.get("markets")
            if not isinstance(markets, list) or not all(isinstance(row, dict) for row in markets):
                raise InvalidResponseError("invalid markets response")
            rows.extend(markets)
            cursor = body.get("cursor", "")
            if not isinstance(cursor, str):
                raise InvalidResponseError("invalid pagination cursor")
            if not cursor:
                break
            if cursor in seen:
                raise InvalidResponseError("repeated pagination cursor")
            seen.add(cursor)
        else:
            raise InvalidResponseError("market pagination exceeded the page budget")
        now = _utc(self._clock())
        candidates: list[ActiveMarket] = []
        for row in rows:
            ticker = row.get("ticker")
            if not isinstance(ticker, str) or not ticker.startswith(f"{series_ticker}-"):
                continue
            if row.get("status") not in {"open", "active"} or row.get("result") not in (None, ""):
                continue
            if "is_provisional" in row and type(row["is_provisional"]) is not bool:
                raise InvalidResponseError("invalid provisional flag")
            market_shard = _index(row.get("exchange_index", 2), "exchange index")
            if row.get("is_provisional") is True or market_shard != self.runtime.exchange_index:
                continue
            if row.get("market_type", "binary") != "binary":
                continue
            opened, closed = _date(row.get("open_time"), "open time"), _date(row.get("close_time"), "close time")
            start = closed - timedelta(minutes=15)
            if not opened <= now < closed or not start <= now:
                continue
            if closed.minute % 15 or closed.second or closed.microsecond:
                continue
            yes_bid, yes_ask = self._quote(row, "yes_bid", ask=False), self._quote(row, "yes_ask", ask=True)
            no_bid, no_ask = self._quote(row, "no_bid", ask=False), self._quote(row, "no_ask", ask=True)
            if no_bid is None and yes_ask is not None:
                no_bid = 100 - yes_ask
            if no_ask is None and yes_bid is not None:
                no_ask = 100 - yes_bid
            for bid, ask in ((yes_bid, yes_ask), (no_bid, no_ask)):
                if bid is not None and ask is not None and bid > ask:
                    raise InvalidResponseError("crossed market quotes")
            strike = None
            if row.get("floor_strike") is not None:
                price = _decimal(row["floor_strike"], "strike price", minimum=Decimal("0.00000001"))
                strike = float(price)
                if not math.isfinite(strike):
                    raise InvalidResponseError("strike overflows float representation")
            event_ticker, strike_type = row.get("event_ticker", ""), row.get("strike_type")
            if not isinstance(event_ticker, str) or strike_type is not None and not isinstance(strike_type, str):
                raise InvalidResponseError("invalid market identifier or strike type")
            candidates.append(ActiveMarket(
                ticker, event_ticker, row["status"], opened, closed, start, now,
                yes_bid, yes_ask, no_bid, no_ask, strike,
                strike_type, _decimal(row["volume_fp"], "volume") if row.get("volume_fp") is not None else None,
                exchange_index=market_shard,
            ))
        return min(candidates, key=lambda market: (market.close_time, market.ticker)) if candidates else None

    def get_orderbook(self, ticker: str) -> OrderBook:
        """Read public fixed-point bids; asks follow the single-book parity."""
        _identifier(ticker, "ticker")
        body = self._request("GET", f"{API_ROOT}/markets/{ticker}/orderbook", authenticated=False)
        fixed = "orderbook_fp" in body
        book = body.get("orderbook_fp" if fixed else "orderbook")
        if not isinstance(book, dict):
            raise InvalidResponseError("missing orderbook")

        def best(side: str) -> tuple[Decimal | None, Decimal]:
            levels = book.get(f"{side}_dollars" if fixed else side)
            if levels is None:
                raise InvalidResponseError("missing orderbook side")
            if not isinstance(levels, list):
                raise InvalidResponseError("invalid orderbook levels")
            by_price: dict[Decimal, Decimal] = {}
            for level in levels:
                if not isinstance(level, list) or len(level) != 2:
                    raise InvalidResponseError("invalid orderbook level")
                price = _decimal(level[0], "book price") / (1 if fixed else 100)
                size = _decimal(level[1], "book size")
                if price > 1 or price in by_price:
                    raise InvalidResponseError("invalid/duplicate book price")
                if size:
                    by_price[price] = size
            price = max(by_price) if by_price else None
            return price, by_price[price] if price is not None else Decimal(0)

        yes, yes_size = best("yes")
        no, no_size = best("no")
        if yes is not None and no is not None and yes + no > 1:
            raise InvalidResponseError("crossed single-book prices")
        return OrderBook(ticker, _utc(self._clock()), yes, yes_size, no, no_size)

    def get_market_lifecycle(self, ticker: str) -> MarketLifecycle:
        """Read an official payout; never infer settlement from Coinbase spot."""
        _identifier(ticker, "ticker")
        row = self._request("GET", f"{API_ROOT}/markets/{ticker}", authenticated=False).get("market")
        if not isinstance(row, dict) or row.get("ticker") != ticker or _index(row.get("exchange_index", self.runtime.exchange_index), "exchange index") != self.runtime.exchange_index:
            raise InvalidResponseError("wrong market/shard in lifecycle response")
        status = row.get("status")
        if not isinstance(status, str):
            raise InvalidResponseError("missing market status")
        payout = None
        if status in {"settled", "finalized"}:
            if row.get("settlement_value_dollars") is not None:
                payout = _decimal(row["settlement_value_dollars"], "settlement payout")
                if payout > 1:
                    raise InvalidResponseError("settlement payout exceeds one dollar")
            elif row.get("result") in {"yes", "no"}:
                payout = Decimal(int(row["result"] == "yes"))
        return MarketLifecycle(ticker, status, _date(row.get("close_time"), "close time"), payout, _utc(self._clock()))

    def _pages(self, path: str, key: str, params: dict[str, str | int]) -> list[dict[str, Any]]:
        """Bound account pagination; reject cursor loops and malformed rows."""
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _ in range(self.policy.max_market_pages):
            body = self._request("GET", path, params=params)
            page = body.get(key)
            if not isinstance(page, list) or not all(isinstance(row, dict) for row in page):
                raise InvalidResponseError(f"invalid {key} response")
            rows.extend(page)
            cursor = body.get("cursor", "")
            if not isinstance(cursor, str) or cursor in seen:
                raise InvalidResponseError("invalid/repeated account cursor")
            if not cursor:
                return rows
            seen.add(cursor)
            params = {**params, "cursor": cursor}
        raise InvalidResponseError("account pagination exceeded page budget")

    def get_series_positions(self, series: str = "KXBTC15M") -> dict[str, Decimal]:
        """Return nonzero signed YES positions, scoped to the configured shard and subaccount 0."""
        _identifier(series, "series")
        rows = self._pages(f"{API_ROOT}/portfolio/positions", "market_positions", {"subaccount": 0, "exchange_index": self.runtime.exchange_index, "limit": 1000})
        positions: dict[str, Decimal] = {}
        for row in rows:
            ticker = row.get("ticker")
            if not isinstance(ticker, str):
                raise InvalidResponseError("missing position ticker")
            if ticker.startswith(series + "-"):
                if _index(row.get("exchange_index", self.runtime.exchange_index), "exchange index") != self.runtime.exchange_index or ticker in positions:
                    raise InvalidResponseError("duplicate/wrong-shard position")
                quantity = _decimal(row.get("position_fp"), "position", minimum=Decimal("-1e24"))
                if quantity:
                    positions[ticker] = quantity
        return positions

    def get_series_orders(self, series: str = "KXBTC15M", *, resting_only: bool = False) -> list[dict[str, Any]]:
        """Read scoped orders for bootstrap and client-ID reconciliation."""
        _identifier(series, "series")
        params: dict[str, str | int] = {"subaccount": 0, "exchange_index": self.runtime.exchange_index, "limit": 1000}
        if resting_only:
            params["status"] = "resting"
        rows = self._pages(f"{API_ROOT}/portfolio/orders", "orders", params)
        selected = []
        for row in rows:
            if not isinstance(row.get("ticker"), str):
                raise InvalidResponseError("missing order ticker")
            if row["ticker"].startswith(series + "-"):
                if _index(row.get("exchange_index", self.runtime.exchange_index), "exchange index") != self.runtime.exchange_index or _index(row.get("subaccount_number", 0), "subaccount") != 0:
                    raise InvalidResponseError("wrong order scope")
                selected.append(row)
        return selected

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch current order state; quantity accounting uses verified fills."""
        _identifier(order_id, "order ID")
        row = self._request("GET", f"{API_ROOT}/portfolio/orders/{order_id}",
                            params={"subaccount": 0, "exchange_index": self.runtime.exchange_index}).get("order")
        if not isinstance(row, dict) or row.get("order_id") != order_id:
            raise InvalidResponseError("wrong order response")
        if _index(row.get("exchange_index", self.runtime.exchange_index), "exchange index") != self.runtime.exchange_index or _index(row.get("subaccount_number", 0), "subaccount") != 0:
            raise InvalidResponseError("wrong order shard/subaccount")
        return row

    def cancel_order(self, order_id: str, *, client_order_id: str) -> None:
        """One cancellation attempt; caller must read the final order and fills."""
        if self.runtime.mode is TradingMode.PAPER:
            raise TradingDisabledError("paper mode cannot cancel exchange orders")
        if self.environment is KalshiEnvironment.PROD and (self.runtime.mode is not TradingMode.LIVE or not self.runtime.live_trading_enabled):
            raise TradingDisabledError("production cancellation requires enabled live mode")
        _identifier(order_id, "order ID")
        _identifier(client_order_id, "client order ID")
        body = self._request("DELETE", f"{ORDER_PATH}/{order_id}", params={"subaccount": 0, "exchange_index": self.runtime.exchange_index}, client_order_id=client_order_id)
        if body.get("order_id") != order_id:
            raise OrderSubmissionUnknown(client_order_id)

    def get_fill_totals(self, order_id: str, ticker: str, side: TargetSide) -> FillTotals:
        """Deduplicate fill IDs and sum owned-leg cash amounts and actual fees."""
        _identifier(order_id, "order ID")
        _identifier(ticker, "ticker")
        if side not in {"YES", "NO"}:
            raise ValueError("invalid side")
        rows = self._pages(f"{API_ROOT}/portfolio/fills", "fills", {"order_id": order_id, "ticker": ticker, "subaccount": 0, "exchange_index": self.runtime.exchange_index, "limit": 1000})
        seen: dict[str, dict[str, Any]] = {}
        quantity = notional = fees = Decimal(0)
        for row in rows:
            fill_id = row.get("fill_id")
            if not isinstance(fill_id, str) or not fill_id or row.get("order_id") != order_id or row.get("ticker", row.get("market_ticker")) != ticker:
                raise InvalidResponseError("wrong fill identity")
            if _index(row.get("exchange_index", self.runtime.exchange_index), "exchange index") != self.runtime.exchange_index or _index(row.get("subaccount_number", 0), "subaccount") != 0:
                raise InvalidResponseError("wrong fill scope")
            if fill_id in seen:
                if seen[fill_id] != row:
                    raise InvalidResponseError("conflicting duplicate fill")
                continue
            seen[fill_id] = row
            count = _decimal(row.get("count_fp"), "fill quantity")
            price = _decimal(row.get(f"{side.lower()}_price_dollars"), "fill price")
            if price > 1:
                raise InvalidResponseError("fill price exceeds one dollar")
            quantity += count
            notional += count * price
            fees += _decimal(row.get("fee_cost"), "fill fee")
        return FillTotals(quantity, notional, fees)

    def send_order(
        self, ticker: str, target_side: TargetSide, is_closing: bool, count: int, price_cents: int,
        *, client_order_id: str | None = None,
    ) -> OrderResult:
        """Submit exactly once with the held/desired leg's own limit price.

        YES buy: bid at p; YES close: ask at p. NO buy: ask at 1-p;
        NO close: bid at 1-p. Close is always reduce-only IOC; entry is GTC.
        A partial IOC acknowledgement is a partial fill, not a closed position.
        Reusing an ID is for caller-controlled reconciliation, NOT a promise
        that repeating an ambiguous request is safe or automatically deduped.
        """
        _identifier(ticker, "ticker")
        if target_side not in {"YES", "NO"}:
            raise ValueError("target_side must be YES or NO")
        if type(is_closing) is not bool:
            raise TypeError("is_closing must be boolean")
        if type(count) is not int or not 1 <= count <= 2_147_483_647:
            raise ValueError("count must be an integer in [1, 2147483647]")
        if type(price_cents) is not int or not 1 <= price_cents <= 99:
            raise ValueError("price_cents must be an integer in [1, 99]")
        if self.runtime.mode is TradingMode.PAPER:
            raise TradingDisabledError("paper mode cannot submit exchange orders")
        if self.environment is KalshiEnvironment.PROD and (
            self.runtime.mode is not TradingMode.LIVE or not self.runtime.live_trading_enabled
        ):
            raise TradingDisabledError("production orders require explicitly enabled live mode")
        if self.environment is KalshiEnvironment.DEMO and self.runtime.mode is not TradingMode.DEMO:
            raise TradingDisabledError("demo orders require demo mode")
        order_id = _identifier(client_order_id if client_order_id is not None else str(self._uuid_factory()), "client order ID")
        yes_price_cents = price_cents if target_side == "YES" else 100 - price_cents
        side = "bid" if (target_side == "YES") != is_closing else "ask"
        payload: dict[str, object] = {
            "ticker": ticker, "client_order_id": order_id, "side": side,
            "count": f"{count}.00", "price": f"{Decimal(yes_price_cents) / 100:.4f}",
            "time_in_force": "immediate_or_cancel" if is_closing else "good_till_canceled",
            "reduce_only": is_closing, "subaccount": 0, "exchange_index": self.runtime.exchange_index,
            "self_trade_prevention_type": "taker_at_cross",
        }
        body = self._request("POST", ORDER_PATH, payload=payload, client_order_id=order_id)
        try:
            exchange_id = body["order_id"]
            if not isinstance(exchange_id, str) or not exchange_id:
                raise InvalidResponseError("missing exchange order ID")
            if body.get("client_order_id", order_id) != order_id:
                raise InvalidResponseError("order correlation mismatch")
            filled = _decimal(body.get("fill_count"), "filled count")
            remaining = _decimal(body.get("remaining_count"), "remaining count")
            if filled + remaining > count:
                raise InvalidResponseError("order quantity exceeds submitted quantity")
            millis = _index(body.get("ts_ms"), "order timestamp")
            processed = datetime.fromtimestamp(millis / 1000, timezone.utc)
            average_price = _decimal(body["average_fill_price"], "average fill price") if body.get("average_fill_price") is not None else None
            if average_price is not None and average_price > 1:
                raise InvalidResponseError("average fill price exceeds one dollar")
            average_fee = _decimal(body["average_fee_paid"], "average fee") if body.get("average_fee_paid") is not None else None
            return OrderResult(exchange_id, order_id, filled, remaining, processed, average_price, average_fee)
        except (InvalidResponseError, KeyError, ValueError, OverflowError, OSError):
            raise OrderSubmissionUnknown(order_id) from None
