"""Verify authentication cryptographically and exercise protocol failures offline."""

import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
import requests
from requests.adapters import HTTPAdapter

from kalshi_ml.config import KalshiEnvironment, RuntimeConfig, Settings, TradingMode
from kalshi_ml.execution.kalshi_client import (
    API_ROOT, ORDER_PATH, APIError, AuthenticationError, ClientPolicy,
    InvalidResponseError, KalshiClient, KalshiError, OrderSubmissionUnknown,
    RateLimitError, TradingDisabledError, TransportError, load_signing_key,
    safe_to_cents,
)


def response(body: object, status: int = 200, **headers: str) -> Mock:
    """A context-managed response that never touches the network."""
    result = Mock(status_code=status, headers=headers)
    result.json.return_value = body
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class KalshiClientTests(unittest.TestCase):
    """Use ephemeral test keys and injected real sessions with mocked transport."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.ed_key = ed25519.Ed25519PrivateKey.generate()

    def setUp(self) -> None:
        self.now = datetime(2026, 11, 1, 12, 5, 0, 123000, tzinfo=timezone.utc)
        self.session = requests.Session()
        self.session.request = Mock()
        self.addCleanup(self.session.close)
        self.sleeps: list[float] = []
        self.order_id = "8c35ecb3-328f-4f52-8c7c-0f4b9862f8d1"
        self.client = self.make_client()

    def make_client(self, **overrides: object) -> KalshiClient:
        values = {
            "environment": KalshiEnvironment.DEMO,
            "key_id": "test-key-id", "private_key": self.ed_key,
            "trading_mode": TradingMode.DEMO,
            "exchange_index": 0,
            "session": self.session, "clock": lambda: self.now,
            "sleep": self.sleeps.append, "jitter": lambda: 0,
            "policy": ClientPolicy(min_request_interval_sec=0),
            "uuid_factory": lambda: uuid.UUID(self.order_id),
        }
        values.update(overrides)
        return KalshiClient(**values)

    def acknowledgement(self, **overrides: object) -> dict[str, object]:
        result = {
            "order_id": "exchange-order", "client_order_id": self.order_id,
            "fill_count": "1.00", "remaining_count": "0.00",
            "ts_ms": int(self.now.timestamp() * 1000),
            "average_fill_price": "0.5600", "average_fee_paid": "0.0200",
        }
        result.update(overrides)
        return result

    def market(self, **overrides: object) -> dict[str, object]:
        result = {
            "ticker": "KXBTC15M-26NOV011215-15", "event_ticker": "KXBTC15M-26NOV011215",
            "status": "active", "market_type": "binary",
            "open_time": "2026-11-01T11:30:00Z", "close_time": "2026-11-01T12:15:00Z",
            "yes_bid_dollars": "0.5400", "yes_ask_dollars": "0.5600",
            "no_bid_dollars": "0.4400", "no_ask_dollars": "0.4600",
            "floor_strike": "100000.25", "strike_type": "greater", "volume_fp": "12.50",
            "exchange_index": "0", "result": "", "is_provisional": False,
        }
        result.update(overrides)
        return result

    def test_exact_price_units_and_conservative_subcents(self) -> None:
        for value in ("0.5600", Decimal("0.56"), 0.56):
            self.assertEqual(safe_to_cents(value), 56)
        self.assertEqual(safe_to_cents("56", unit="cents"), 56)
        self.assertEqual(safe_to_cents("1"), 100)
        self.assertEqual(safe_to_cents("1", unit="cents"), 1)
        self.assertEqual(safe_to_cents("0.5550", rounding="floor"), 55)
        self.assertEqual(safe_to_cents("0.5550", rounding="ceil"), 56)
        for value in (None, True, "", "NaN", "Infinity", "-0.1", "1.01", "0.5550", "1e999999"):
            with self.subTest(value=value), self.assertRaises(InvalidResponseError):
                safe_to_cents(value)

    def test_rsa_pss_and_ed25519_signatures_verify(self) -> None:
        for key in (self.rsa_key, self.ed_key):
            client = self.make_client(private_key=key)
            headers = client._auth_headers("get", API_ROOT + "/portfolio/balance?subaccount=0")
            timestamp = str(int(self.now.timestamp() * 1000))
            self.assertEqual(headers["KALSHI-ACCESS-TIMESTAMP"], timestamp)
            self.assertEqual(headers["KALSHI-ACCESS-KEY"], "test-key-id")
            message = f"{timestamp}GET{API_ROOT}/portfolio/balance".encode()
            signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"], validate=True)
            if isinstance(key, rsa.RSAPrivateKey):
                key.public_key().verify(signature, message,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
            else:
                key.public_key().verify(signature, message)

    def test_key_files_encryption_and_settings_factory(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ephemeral.pem"
            for key in (self.rsa_key, self.ed_key):
                path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                   serialization.BestAvailableEncryption(b"test-pass")))
                loaded = load_signing_key(path, password=b"test-pass")
                self.assertIsInstance(loaded, type(key))
                with self.assertRaises(AuthenticationError):
                    load_signing_key(path)
                settings = Settings(key_id="test-key-id", private_key_path=path, runtime=RuntimeConfig(mode=TradingMode.DEMO))
                client = KalshiClient.from_settings(settings, password=b"test-pass", session=self.session)
                self.assertEqual(client.runtime.mode, TradingMode.DEMO)
        self.session.request.assert_not_called()

    def test_unsupported_and_small_keys_rejected(self) -> None:
        for key in (ec.generate_private_key(ec.SECP256R1()), rsa.generate_private_key(public_exponent=65537, key_size=1024)):
            with self.assertRaises(AuthenticationError):
                self.make_client(private_key=key)

    def test_hosts_and_runtime_guards(self) -> None:
        for environment, mode, enabled, host in (
            (KalshiEnvironment.DEMO, TradingMode.DEMO, False, "https://external-api.demo.kalshi.co"),
            (KalshiEnvironment.PROD, TradingMode.LIVE, True, "https://external-api.kalshi.com"),
        ):
            client = self.make_client(environment=environment, trading_mode=mode, live_trading_enabled=enabled)
            self.session.request.return_value = response({"balance_dollars": "100.00"})
            client.get_balance()
            self.assertEqual(self.session.request.call_args.args[1], host + API_ROOT + "/portfolio/balance")
        for values in (
            {"environment": KalshiEnvironment.PROD},
            {"trading_mode": TradingMode.LIVE},
            {"environment": KalshiEnvironment.PROD, "trading_mode": TradingMode.LIVE},
        ):
            with self.assertRaises(ValueError):
                self.make_client(**values)

    def test_balance_selects_shard_zero_and_preserves_subcents(self) -> None:
        self.session.request.return_value = response({
            "balance": 999999, "balance_dollars": "9999.99", "portfolio_value": 123,
            "updated_ts": int(self.now.timestamp()),
            "balance_breakdown": [{"exchange_index": 2, "balance": "8000.00"}, {"exchange_index": "0", "balance": "10.123456"}],
        })
        balance = self.client.get_balance()
        self.assertEqual(balance.balance_dollars, Decimal("10.123456"))
        self.assertEqual(balance.balance_cents, Decimal("1012.3456"))
        self.assertEqual(balance.portfolio_value_dollars, Decimal("1.23"))
        self.assertEqual(self.session.request.call_args.kwargs["params"], {"subaccount": 0, "exchange_index": 0})

    def test_scoped_balance_fallback_and_missing_shard_fail_closed(self) -> None:
        for body, expected in (({"balance_dollars": "10.25", "balance": 999}, "10.25"), ({"balance": 1025}, "10.25")):
            self.session.request.return_value = response(body)
            self.assertEqual(self.client.get_balance().balance_dollars, Decimal(expected))
        for breakdown in ([], [{"exchange_index": 1, "balance": "20"}],
                          [{"exchange_index": 0, "balance": "20"}] * 2, None):
            self.session.request.return_value = response({"balance_dollars": "99999", "balance_breakdown": breakdown})
            with self.assertRaises(InvalidResponseError):
                self.client.get_balance()

    def test_market_pagination_rejects_future_and_expired_cycles(self) -> None:
        future = self.market(ticker="KXBTC15M-FUTURE", close_time="2026-11-01T12:30:00Z")
        expired = self.market(ticker="KXBTC15M-EXPIRED", close_time="2026-11-01T12:00:00Z")
        paused = self.market(status="paused")
        self.session.request.side_effect = [response({"markets": [future, expired, paused], "cursor": "next"}),
                                            response({"markets": [self.market()], "cursor": ""})]
        market = self.client.get_active_15m_market()
        self.assertIsNotNone(market)
        self.assertEqual(market.strike_price, 100000.25)
        self.assertEqual(market.yes_ask_cents, 56)
        self.assertEqual(market.no_ask_cents, 46)
        self.assertEqual(market.cycle_start_ts.minute, 0)
        self.assertEqual(market.volume, Decimal("12.50"))
        self.assertEqual(self.session.request.call_args.kwargs["params"]["cursor"], "next")
        self.assertNotIn("KALSHI-ACCESS-KEY", self.session.request.call_args.kwargs["headers"])

    def test_provisional_other_shard_and_not_yet_open_are_not_active(self) -> None:
        for row in (self.market(is_provisional=True), self.market(exchange_index=1),
                    self.market(open_time="2026-11-01T12:06:00Z"), self.market(result="yes"),
                    self.market(close_time="2026-11-01T12:16:00Z"), self.market(ticker="OTHER-1")):
            self.session.request.return_value = response({"markets": [row]})
            self.assertIsNone(self.client.get_active_15m_market())

    def test_market_missing_numbers_remain_none_and_subcents_round_safely(self) -> None:
        row = self.market(yes_bid_dollars="0.5550", yes_ask_dollars="0.5650", floor_strike=None)
        row.pop("no_bid_dollars")
        row.pop("no_ask_dollars")
        self.session.request.return_value = response({"markets": [row]})
        market = self.client.get_active_15m_market()
        self.assertEqual((market.yes_bid_cents, market.yes_ask_cents, market.no_bid_cents, market.no_ask_cents), (55, 57, 43, 45))
        self.assertIsNone(market.strike_price)
        for field in ("yes_bid_dollars", "yes_ask_dollars"):
            row.pop(field)
        self.session.request.return_value = response({"markets": [row]})
        self.assertIsNone(self.client.get_active_15m_market().yes_ask_cents)

    def test_malformed_quotes_and_repeated_cursor_fail_closed(self) -> None:
        self.session.request.return_value = response({"markets": [self.market(yes_ask_dollars="NaN")]})
        with self.assertRaises(InvalidResponseError):
            self.client.get_active_15m_market()
        self.session.request.return_value = response({"markets": [], "cursor": "loop"})
        with self.assertRaises(InvalidResponseError):
            self.client.get_active_15m_market()

    def test_four_order_parities_and_required_wire_types(self) -> None:
        for side, closing, book_side, price in (
            ("YES", False, "bid", "0.5600"), ("NO", False, "ask", "0.4400"),
            ("YES", True, "ask", "0.5600"), ("NO", True, "bid", "0.4400"),
        ):
            self.session.request.return_value = response(self.acknowledgement(fill_count="1.00", remaining_count="0.00"), 201)
            result = self.client.send_order("KXBTC15M-TEST", side, closing, 1, 56)
            call = self.session.request.call_args
            self.assertEqual(call.args, ("POST", "https://external-api.demo.kalshi.co" + ORDER_PATH))
            payload = call.kwargs["json"]
            self.assertEqual(payload["side"], book_side)
            self.assertEqual(payload["price"], price)
            self.assertEqual(payload["count"], "1.00")
            self.assertEqual(payload["reduce_only"], closing)
            self.assertEqual(payload["time_in_force"], "immediate_or_cancel" if closing else "good_till_canceled")
            self.assertEqual(payload["subaccount"], 0)
            self.assertEqual(payload["exchange_index"], 0)
            self.assertEqual(payload["self_trade_prevention_type"], "taker_at_cross")
            self.assertEqual(result.fill_count, Decimal("1.00"))
            self.assertEqual(result.average_fee_paid_dollars, Decimal("0.02"))
            self.assertFalse(call.kwargs["allow_redirects"])

    def test_partial_ioc_is_not_mistaken_for_a_closed_position(self) -> None:
        self.session.request.return_value = response(self.acknowledgement(fill_count="0.25"), 201)
        result = self.client.send_order("KXBTC15M-TEST", "YES", True, 1, 56)
        self.assertEqual(result.fill_count, Decimal("0.25"))
        self.assertEqual(result.remaining_count, 0)

    def test_order_failures_are_never_blindly_retried(self) -> None:
        failures = (requests.Timeout(), requests.ConnectionError(), response({}, 500), response({}, 503),
                    response({}, 408), response({}, 409), response({}, 302), response({}, 200),
                    response({}, 201), response(self.acknowledgement(client_order_id="other"), 201))
        for failure in failures:
            self.session.request.reset_mock()
            self.session.request.side_effect = [failure]
            with self.subTest(failure=failure), self.assertRaises(OrderSubmissionUnknown) as raised:
                self.client.send_order("KXBTC15M-TEST", "NO", False, 1, 56)
            self.assertEqual(raised.exception.client_order_id, self.order_id)
            self.assertEqual(self.session.request.call_count, 1)

    def test_order_rate_limit_contains_id_and_does_not_retry(self) -> None:
        self.session.request.return_value = response({}, 429, **{"Retry-After": "2"})
        with self.assertRaises(RateLimitError) as raised:
            self.client.send_order("KXBTC15M-TEST", "YES", False, 1, 56)
        self.assertEqual(raised.exception.client_order_id, self.order_id)
        self.assertEqual(raised.exception.retry_after_sec, 2)
        self.assertEqual(self.session.request.call_count, 1)

    def test_invalid_orders_and_paper_mode_make_no_requests(self) -> None:
        for side, closing, count, price in (("INVALID", False, 1, 56), ("YES", 1, 1, 56),
                                           ("YES", False, True, 56), ("YES", False, 0, 56),
                                           ("YES", False, 1, 0), ("YES", False, 1, 100)):
            with self.assertRaises((ValueError, TypeError)):
                self.client.send_order("KXBTC15M-TEST", side, closing, count, price)
        with self.assertRaises(TradingDisabledError):
            self.make_client(trading_mode=TradingMode.PAPER).send_order("KXBTC15M-TEST", "YES", False, 1, 56)
        self.session.request.assert_not_called()

    def test_get_backoff_and_fresh_signatures_each_attempt(self) -> None:
        times = iter((self.now, self.now + timedelta(seconds=2)))
        client = self.make_client(clock=lambda: next(times))
        self.session.request.side_effect = [response({}, 429, **{"Retry-After": "2"}), response({"balance_dollars": "10"})]
        self.assertEqual(client.get_balance().balance_dollars, 10)
        headers = [call.kwargs["headers"] for call in self.session.request.call_args_list]
        self.assertNotEqual(headers[0]["KALSHI-ACCESS-TIMESTAMP"], headers[1]["KALSHI-ACCESS-TIMESTAMP"])
        self.assertIn(2, self.sleeps)

    def test_get_timeout_budget_and_long_retry_after(self) -> None:
        self.session.request.side_effect = requests.Timeout()
        with self.assertRaises(TransportError):
            self.client.get_balance()
        self.assertEqual(self.session.request.call_count, 4)
        self.session.request.reset_mock(side_effect=True)
        self.session.request.return_value = response({}, 429, **{"Retry-After": "600"})
        with self.assertRaises(RateLimitError):
            self.client.get_balance()
        self.assertEqual(self.session.request.call_count, 1)

    def test_get_auth_redirect_and_malformed_json_errors(self) -> None:
        for status, expected in ((401, AuthenticationError), (403, AuthenticationError), (302, APIError), (400, APIError)):
            self.session.request.return_value = response({"message": "must-not-be-logged"}, status)
            with self.assertRaises(expected) as raised:
                self.client.get_balance()
            self.assertNotIn("must-not-be-logged", str(raised.exception))
        malformed = response({})
        malformed.json.side_effect = ValueError("invalid JSON")
        self.session.request.return_value = malformed
        with self.assertRaises(InvalidResponseError):
            self.client.get_balance()

    def test_public_discovery_needs_no_key_but_balance_does(self) -> None:
        client = KalshiClient(session=self.session, clock=lambda: self.now)
        self.session.request.return_value = response({"markets": []})
        self.assertIsNone(client.get_active_15m_market())
        with self.assertRaises(AuthenticationError):
            client.get_balance()

    def test_adapter_retries_and_signature_destination_are_rejected(self) -> None:
        self.session.mount("https://", HTTPAdapter(max_retries=2))
        with self.assertRaises(ValueError):
            self.make_client()
        self.session.mount("https://", HTTPAdapter(max_retries=0))
        with self.assertRaises(ValueError):
            self.client._auth_headers("GET", "https://example.com" + API_ROOT + "/portfolio/balance")
        with self.assertRaises(ValueError):
            self.client._request("GET", API_ROOT + "/portfolio/balance?subaccount=0")

    def test_session_ownership_and_context_lifecycle(self) -> None:
        with self.client:
            pass
        with self.assertRaises(KalshiError):
            self.client.get_balance()
        with self.assertRaises(KalshiError):
            self.client.__enter__()
        with patch("kalshi_ml.execution.kalshi_client.requests.Session") as factory:
            owned = factory.return_value
            owned.adapters = {}
            with KalshiClient():
                pass
            owned.close.assert_called_once()

    def test_orderbook_fixed_point_depth_and_complementary_asks(self) -> None:
        self.session.request.return_value = response({"orderbook_fp": {
            "yes_dollars": [["0.5400", "1.50"], ["0.1000", "5.00"]],
            "no_dollars": [["0.4400", "2.25"]],
        }})
        book = self.client.get_orderbook("KXBTC15M-TEST")
        self.assertEqual(book.quote("YES", closing=False), (Decimal(".56"), Decimal("2.25")))
        self.assertEqual(book.quote("NO", closing=True), (Decimal(".44"), Decimal("2.25")))
        self.assertEqual(book.quote("YES", closing=True), (Decimal(".54"), Decimal("1.50")))
        self.assertEqual(book.quote("NO", closing=False), (Decimal(".46"), Decimal("1.50")))
        self.assertNotIn("KALSHI-ACCESS-KEY", self.session.request.call_args.kwargs["headers"])

    def test_empty_legacy_and_crossed_orderbooks(self) -> None:
        self.session.request.return_value = response({"orderbook": {"yes": [[54, 2]], "no": []}})
        book = self.client.get_orderbook("KXBTC15M-TEST")
        self.assertEqual(book.yes_bid, Decimal(".54"))
        self.assertEqual(book.quote("YES", closing=False), (None, Decimal(0)))
        self.session.request.return_value = response({"orderbook_fp": {"yes_dollars": [[".7", "1"]], "no_dollars": [[".6", "1"]]}})
        with self.assertRaises(InvalidResponseError):
            self.client.get_orderbook("KXBTC15M-TEST")

    def test_official_settlement_waits_until_final_status(self) -> None:
        for status, expected in (("closed", None), ("determined", None), ("finalized", Decimal(".5"))):
            self.session.request.return_value = response({"market": self.market(
                status=status, result="yes", settlement_value_dollars="0.5000")})
            result = self.client.get_market_lifecycle(self.market()["ticker"])
            self.assertEqual(result.yes_payout, expected)
        self.session.request.return_value = response({"market": self.market(status="finalized", exchange_index=2)})
        with self.assertRaises(InvalidResponseError):
            self.client.get_market_lifecycle(self.market()["ticker"])

    def test_fill_aggregation_deduplicates_and_uses_owned_leg_prices(self) -> None:
        fill = {"fill_id": "fill-1", "order_id": "order-1", "ticker": "KXBTC15M-TEST",
                "count_fp": "0.50", "yes_price_dollars": "0.5600", "no_price_dollars": "0.4400", "fee_cost": "0.0100"}
        self.session.request.side_effect = [response({"fills": [fill], "cursor": "next"}),
                                            response({"fills": [fill], "cursor": ""})]
        totals = self.client.get_fill_totals("order-1", "KXBTC15M-TEST", "NO")
        self.assertEqual(totals.quantity, Decimal(".5"))
        self.assertEqual(totals.notional, Decimal(".22"))
        self.assertEqual(totals.fees, Decimal(".01"))

    def test_cancel_is_single_attempt_and_paper_cannot_cancel(self) -> None:
        self.session.request.return_value = response({"order_id": "order-1"})
        self.client.cancel_order("order-1", client_order_id=self.order_id)
        self.assertEqual(self.session.request.call_args.args[0], "DELETE")
        self.assertEqual(self.session.request.call_args.kwargs["params"], {"subaccount": 0, "exchange_index": 0})
        self.session.request.reset_mock()
        self.session.request.side_effect = requests.Timeout()
        with self.assertRaises(OrderSubmissionUnknown):
            self.client.cancel_order("order-1", client_order_id=self.order_id)
        self.assertEqual(self.session.request.call_count, 1)
        with self.assertRaises(TradingDisabledError):
            self.make_client(trading_mode=TradingMode.PAPER).cancel_order("order-1", client_order_id=self.order_id)

    def test_signed_positions_include_no_and_orders_validate_scope(self) -> None:
        self.session.request.return_value = response({"market_positions": [
            {"ticker": "KXBTC15M-TEST", "position_fp": "-0.50", "exchange_index": 0},
        ], "cursor": ""})
        self.assertEqual(self.client.get_series_positions(), {"KXBTC15M-TEST": Decimal("-.5")})
        self.session.request.return_value = response({"orders": [{"ticker": "KXBTC15M-TEST", "exchange_index": 2}]})
        with self.assertRaises(InvalidResponseError):
            self.client.get_series_orders()

    def test_crypto_shard_is_returned_from_market_response(self) -> None:
        client = self.make_client(exchange_index=2)
        self.session.request.return_value = response({"markets": [self.market(exchange_index="2")]})
        market = client.get_active_15m_market()
        self.assertEqual(market.exchange_index, 2)
        row = self.market()
        row.pop("exchange_index")
        self.session.request.return_value = response({"markets": [row]})
        self.assertEqual(client.get_active_15m_market().exchange_index, 2)

    def test_crypto_shard_routes_balance_orders_fills_and_cancellation(self) -> None:
        client = self.make_client(exchange_index=2)
        self.session.request.return_value = response({"balance_breakdown": [
            {"exchange_index": 0, "balance": "9999.00"}, {"exchange_index": 2, "balance": "12.34"},
        ]})
        balance = client.get_balance()
        self.assertEqual(balance.exchange_index, 2)
        self.assertEqual(balance.balance_dollars, Decimal("12.34"))
        self.assertEqual(self.session.request.call_args.kwargs["params"]["exchange_index"], 2)
        self.session.request.return_value = response(self.acknowledgement(), 201)
        client.send_order("KXBTC15M-TEST", "YES", False, 1, 56)
        self.assertEqual(self.session.request.call_args.kwargs["json"]["exchange_index"], 2)
        self.session.request.return_value = response({"order_id": "order-1"})
        client.cancel_order("order-1", client_order_id=self.order_id)
        self.assertEqual(self.session.request.call_args.kwargs["params"]["exchange_index"], 2)
        self.session.request.return_value = response({"order": {"order_id": "order-1", "exchange_index": 2}})
        client.get_order("order-1")
        self.assertEqual(self.session.request.call_args.kwargs["params"]["exchange_index"], 2)
        self.session.request.return_value = response({"fills": []})
        client.get_fill_totals("order-1", "KXBTC15M-TEST", "YES")
        self.assertEqual(self.session.request.call_args.kwargs["params"]["exchange_index"], 2)


if __name__ == "__main__":
    unittest.main()
