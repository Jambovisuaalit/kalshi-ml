"""Module 1: validated configuration for the KXBTC15M trading strategy.

Prices are expressed in whole cents and spot-price differences in USD.
Configuration does not authenticate, submit orders, or enable live trading.
Use a separate execution adapter for exchange-specific price serialization.
"""

from __future__ import annotations

from dataclasses import field, fields
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
import os
from collections.abc import Mapping
from typing import Final

from dotenv import dotenv_values
from pydantic import ConfigDict, SecretStr
from pydantic.dataclasses import dataclass

VALIDATION = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True, validate_default=True)


FEATURE_NAMES: Final[tuple[str, ...]] = (
    "delta_usd",
    "remaining_sec",
    "z_score",
    "vol_15m",
    "vol_60m",
    "vol_ratio",
    "mom_1m",
    "mom_3m",
    "obi",
)
"""Ordered model input schema; training and inference must use this order."""

SERIES_TICKER: Final[str] = "KXBTC15M"
CONTRACT_DURATION_SEC: Final[int] = 900


class TradingMode(StrEnum):
    """Explicit execution environments; paper mode is the safe default."""

    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


def _require_int(name: str, value: int, minimum: int, maximum: int) -> None:
    """Validate integer bounds while rejecting booleans and implicit coercion."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _require_decimal(name: str, value: Decimal) -> None:
    """Reject floats, infinities, and NaNs in monetary and probability values."""
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True, config=VALIDATION)
class StrategyConfig:
    """Immutable signal and exit thresholds.

    Entry bounds are inclusive. YES requires probability >= its threshold
    and delta >= minimum_delta_usd. NO requires probability <= its threshold
    and delta <= -minimum_delta_usd. Both require the corresponding ask to
    fall inside the entry-price interval.

    Exit rules are independent triggers evaluated against executable bids:
    take profit at >= take_profit_cents; stop at a loss of >= hard_stop_cents;
    or exit immediately when delta <= 0 for YES or delta >= 0 for NO.
    These price thresholds exclude fees; risk accounting must include fees.
    """

    entry_min_remaining_sec: int = 180
    entry_max_remaining_sec: int = 800
    yes_probability_min: Decimal = Decimal("0.55")
    no_probability_max: Decimal = Decimal("0.45")
    minimum_delta_usd: Decimal = Decimal("5.0")
    entry_min_cents: int = 45
    entry_max_cents: int = 66
    take_profit_cents: int = 88
    hard_stop_cents: int = 14
    volatility_sigma_limit: Decimal = Decimal("2")
    minimum_edge_cents: Decimal = Decimal("1")
    entry_fee_buffer_cents: Decimal = Decimal("2")
    exit_fee_buffer_cents: Decimal = Decimal("2")

    def __post_init__(self) -> None:
        """Fail at construction when thresholds are invalid or contradictory."""
        for name in ("volatility_sigma_limit", "minimum_edge_cents", "entry_fee_buffer_cents", "exit_fee_buffer_cents"):
            value = getattr(self, name)
            _require_decimal(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.volatility_sigma_limit > 2:
            raise ValueError("volatility_sigma_limit may tighten, but not exceed two sigma")
        for name in ("entry_min_remaining_sec", "entry_max_remaining_sec"):
            _require_int(name, getattr(self, name), 1, CONTRACT_DURATION_SEC - 1)
        if self.entry_min_remaining_sec > self.entry_max_remaining_sec:
            raise ValueError("entry time bounds must be ordered")
        for name in ("yes_probability_min", "no_probability_max"):
            value = getattr(self, name)
            _require_decimal(name, value)
            if not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if not self.no_probability_max < Decimal("0.5") < self.yes_probability_min:
            raise ValueError("probability thresholds must straddle 0.5 strictly")
        _require_decimal("minimum_delta_usd", self.minimum_delta_usd)
        if self.minimum_delta_usd <= 0:
            raise ValueError("minimum_delta_usd must be positive")
        for name in (
            "entry_min_cents", "entry_max_cents", "take_profit_cents",
            "hard_stop_cents",
        ):
            _require_int(name, getattr(self, name), 1, 99)
        if not self.entry_min_cents <= self.entry_max_cents < self.take_profit_cents:
            raise ValueError("entry prices must be ordered below take profit")
        if self.hard_stop_cents >= self.entry_min_cents:
            raise ValueError("hard stop must be smaller than minimum entry price")


@dataclass(frozen=True, slots=True, config=VALIDATION)
class RuntimeConfig:
    """Execution identity with explicit opt-in for live trading.

    This opt-in is necessary but insufficient for live execution: the future
    execution module must also enforce account-level exposure limits, stale
    data checks, reconciliation, and emergency shutdown controls.
    Credentials belong in environment variables or a secret store.
    """

    mode: TradingMode = TradingMode.PAPER
    live_trading_enabled: bool = False
    subaccount: int = 0
    exchange_index: int = 2

    def __post_init__(self) -> None:
        """Enforce the requested account identifiers and live-mode interlock."""
        if not isinstance(self.mode, TradingMode):
            raise TypeError("mode must be a TradingMode")
        if type(self.live_trading_enabled) is not bool:
            raise TypeError("live_trading_enabled must be a boolean")
        _require_int("subaccount", self.subaccount, 0, 0)
        _require_int("exchange_index", self.exchange_index, 0, 2_147_483_647)
        if self.mode is TradingMode.LIVE and not self.live_trading_enabled:
            raise ValueError("live trading requires explicit enablement")
        if self.mode is not TradingMode.LIVE and self.live_trading_enabled:
            raise ValueError("live enablement is only valid in live mode")


class KalshiEnvironment(StrEnum):
    """Exchange credential environment, separate from execution mode."""

    DEMO = "DEMO"
    PROD = "PROD"


@dataclass(frozen=True, slots=True, config=VALIDATION)
class Settings:
    """Application settings loaded explicitly, with no import-time I/O.

    Relative key paths resolve against the dotenv file's parent directory.
    Credentials are excluded from repr and never included in CLI output.
    """

    environment: KalshiEnvironment = KalshiEnvironment.DEMO
    key_id: str = field(default="", repr=False)
    private_key_path: Path | None = field(default=None, repr=False)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    market_poll_sec: Decimal = Decimal("1.0")
    position_poll_sec: Decimal = Decimal("0.5")
    http_timeout_sec: Decimal = Decimal("10.0")
    http_max_retries: int = 3
    alerts_enabled: bool = False
    telegram_bot_token: SecretStr = field(default_factory=lambda: SecretStr(""), repr=False)
    telegram_chat_id: str = field(default="", repr=False)
    discord_webhook_url: SecretStr = field(default_factory=lambda: SecretStr(""), repr=False)

    def __post_init__(self) -> None:
        """Validate polling intervals and environment/mode consistency."""
        if not isinstance(self.environment, KalshiEnvironment):
            raise TypeError("environment must be a KalshiEnvironment")
        if not isinstance(self.strategy, StrategyConfig):
            raise TypeError("strategy must be a StrategyConfig")
        if not isinstance(self.runtime, RuntimeConfig):
            raise TypeError("runtime must be a RuntimeConfig")
        if not isinstance(self.key_id, str):
            raise TypeError("key_id must be a string")
        if self.private_key_path is not None and not isinstance(self.private_key_path, Path):
            raise TypeError("private_key_path must be a Path or None")
        for name in ("market_poll_sec", "position_poll_sec", "http_timeout_sec"):
            value = getattr(self, name)
            _require_decimal(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        _require_int("http_max_retries", self.http_max_retries, 0, 10)
        if self.runtime.mode is TradingMode.DEMO and self.environment is not KalshiEnvironment.DEMO:
            raise ValueError("demo trading requires the DEMO environment")
        if self.runtime.mode is TradingMode.LIVE and self.environment is not KalshiEnvironment.PROD:
            raise ValueError("live trading requires the PROD environment")

    def validate_credentials(self) -> None:
        """Check credentials locally without reading or displaying key contents.

        The exchange adapter separately parses and validates the RSA/Ed25519 key.
        This method makes no network requests and submits no orders.
        """
        if not self.key_id.strip():
            raise ValueError("KALSHI_KEY_ID is required")
        if self.private_key_path is None or not self.private_key_path.is_file():
            raise ValueError("KALSHI_PRIVATE_KEY_PATH must identify an existing file")


def _boolean(name: str, value: str) -> bool:
    """Parse explicit booleans; reject misspellings instead of guessing."""
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return normalized == "true"


def load_settings(
    env_file: str | Path = ".env",
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load defaults < dotenv file < process environment without mutating os.environ.

    Pass an explicit mapping to isolate tests from the process environment.
    A missing dotenv file is allowed; credentials are checked only on request.
    Empty values for numeric settings are invalid and never silently defaulted.
    """
    path = Path(env_file).expanduser().resolve()
    values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    values.update(os.environ if environ is None else environ)
    defaults = StrategyConfig()
    strategy_values: dict[str, int | Decimal] = {}
    for item in fields(defaults):
        name = f"STRATEGY_{item.name.upper()}"
        if name in values:
            converter = Decimal if isinstance(getattr(defaults, item.name), Decimal) else int
            try:
                strategy_values[item.name] = converter(values[name])
            except (ValueError, ArithmeticError) as exc:
                raise ValueError(f"invalid numeric setting: {name}") from exc
    key_path = values.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    resolved_key: Path | None = None
    if key_path:
        resolved_key = Path(key_path).expanduser()
        if not resolved_key.is_absolute():
            resolved_key = path.parent / resolved_key
        resolved_key = resolved_key.resolve()
    return Settings(
        environment=KalshiEnvironment(values.get("KALSHI_ENV", "DEMO").strip().upper()),
        key_id=values.get("KALSHI_KEY_ID", "").strip(),
        private_key_path=resolved_key,
        strategy=StrategyConfig(**strategy_values),
        runtime=RuntimeConfig(
            mode=TradingMode(values.get("KALSHI_TRADING_MODE", "paper").strip().lower()),
            live_trading_enabled=_boolean(
                "KALSHI_LIVE_TRADING_ENABLED", values.get("KALSHI_LIVE_TRADING_ENABLED", "false")
            ),
            subaccount=int(values.get("KALSHI_SUBACCOUNT", "0")),
            exchange_index=int(values.get("KALSHI_EXCHANGE_INDEX", "2")),
        ),
        market_poll_sec=Decimal(values.get("MARKET_POLL_SEC", "1.0")),
        position_poll_sec=Decimal(values.get("POSITION_POLL_SEC", "0.5")),
        http_timeout_sec=Decimal(values.get("HTTP_TIMEOUT_SEC", "10.0")),
        http_max_retries=int(values.get("HTTP_MAX_RETRIES", "3")),
        alerts_enabled=_boolean("ALERTS_ENABLED", values.get("ALERTS_ENABLED", "false")),
        telegram_bot_token=SecretStr(values.get("TELEGRAM_BOT_TOKEN", "")),
        telegram_chat_id=values.get("TELEGRAM_CHAT_ID", "").strip(),
        discord_webhook_url=SecretStr(values.get("DISCORD_WEBHOOK_URL", "")),
    )
