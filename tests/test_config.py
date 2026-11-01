"""Configuration boundaries and credential isolation using the standard library."""

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kalshi_ml.config import Settings, StrategyConfig, load_settings


class ConfigurationTests(unittest.TestCase):
    """Exercise settings ingestion without contacting external services."""

    def test_defaults_without_dotenv(self) -> None:
        """Paper settings must work without credentials or a dotenv file."""
        with TemporaryDirectory() as directory:
            settings = load_settings(Path(directory) / ".env", environ={})
        self.assertEqual(settings.strategy, StrategyConfig())
        self.assertEqual(settings.market_poll_sec, Decimal("1.0"))
        self.assertEqual(settings.runtime.mode.value, "paper")
        self.assertEqual(settings.runtime.exchange_index, 2)

    def test_exchange_index_is_configurable(self) -> None:
        with TemporaryDirectory() as directory:
            for index in (0, 2):
                settings = load_settings(Path(directory) / ".env", environ={"KALSHI_EXCHANGE_INDEX": str(index)})
                self.assertEqual(settings.runtime.exchange_index, index)
            with self.assertRaises(ValueError):
                load_settings(Path(directory) / ".env", environ={"KALSHI_EXCHANGE_INDEX": "-1"})

    def test_environment_precedence_and_relative_key(self) -> None:
        """Process values override dotenv while key paths remain file-relative."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "STRATEGY_ENTRY_MAX_CENTS=60\n"
                "KALSHI_PRIVATE_KEY_PATH=key.pem\nKALSHI_KEY_ID=hidden-id\n",
                encoding="utf-8",
            )
            settings = load_settings(path, environ={"STRATEGY_ENTRY_MAX_CENTS": "65"})
            self.assertEqual(settings.strategy.entry_max_cents, 65)
            self.assertEqual(settings.private_key_path, (Path(directory) / "key.pem").resolve())
            self.assertNotIn("hidden-id", repr(settings))
            self.assertNotIn("key.pem", repr(settings))

    def test_invalid_environment_values(self) -> None:
        """Reject unsafe mode combinations, nonfinite numbers and invalid bounds."""
        invalid = (
            {"KALSHI_TRADING_MODE": "live"},
            {"KALSHI_TRADING_MODE": "live", "KALSHI_LIVE_TRADING_ENABLED": "true"},
            {"KALSHI_TRADING_MODE": "demo", "KALSHI_ENV": "PROD"},
            {"KALSHI_LIVE_TRADING_ENABLED": "tru"},
            {"KALSHI_SUBACCOUNT": "1"},
            {"MARKET_POLL_SEC": "NaN"},
            {"HTTP_TIMEOUT_SEC": "0"},
            {"STRATEGY_ENTRY_MIN_CENTS": "67"},
            {"STRATEGY_MINIMUM_DELTA_USD": "NaN"},
            {"STRATEGY_YES_PROBABILITY_MIN": ""},
        )
        with TemporaryDirectory() as directory:
            for values in invalid:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    load_settings(Path(directory) / ".env", environ=values)

    def test_credentials_are_explicit(self) -> None:
        """Public config validation never requires an API credential."""
        with self.assertRaises(ValueError):
            Settings().validate_credentials()


if __name__ == "__main__":
    unittest.main()
