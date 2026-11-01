"""Distribution, safe bootstrap, alerts and exact collateral units, offline."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import uuid
import zipfile

from cryptography.hazmat.primitives.asymmetric import ed25519
import numpy as np
import pandas as pd
from pydantic import SecretStr, TypeAdapter, ValidationError
import requests

from kalshi_ml.config import FEATURE_NAMES, KalshiEnvironment, Settings, TradingMode
from kalshi_ml.execution import collateral
from kalshi_ml.execution.kalshi_client import KalshiClient, OrderSubmissionUnknown
from kalshi_ml.model.inference import RealtimePredictor
from kalshi_ml.monitoring.notifier import Notifier
from kalshi_ml.setup import initialize
from kalshi_ml.storage.journal import Phase, RuntimeState, StateStore, read_performance
from scripts.build_release import DEPLOY_FILES, MODEL_FILES, ROOT_FILES, build_release


def http_response(payload: object, status: int = 200) -> Mock:
    result = Mock(status_code=status, headers={})
    result.json.return_value = payload
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class DistributionTests(unittest.TestCase):
    """Outbound HTTP is always mocked; transfer scripts run only in dry-run."""

    def test_settings_use_strict_pydantic_validation_and_hide_secrets(self) -> None:
        self.assertIn("properties", TypeAdapter(Settings).json_schema())
        with self.assertRaises(ValidationError):
            Settings(http_max_retries="3")
        settings = Settings(telegram_bot_token=SecretStr("123:test_secret"))
        self.assertNotIn("test_secret", repr(settings))

    def test_inference_schema_and_probability_validation(self) -> None:
        model = Mock()
        model.classes_ = np.array([0, 1])
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
        predictor = RealtimePredictor(model)
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        self.assertEqual(predictor.predict_yes(features), 0.8)
        with self.assertRaises(ValueError):
            predictor.predict_yes({"delta_usd": 1.0})
        invalid = {**features, "vol_15m": float("nan")}
        with self.assertRaises(ValueError):
            predictor.predict_yes(invalid)
        model.predict_proba.return_value = np.array([[0.8, 0.8]])
        with self.assertRaises(ValueError):
            predictor.predict_yes(features)

    def test_inference_rejects_pretraining_decision_time(self) -> None:
        model = Mock()
        predictor = RealtimePredictor(model, {"inference_not_before": "2026-11-01T12:00:00Z"})
        with self.assertRaises(ValueError):
            predictor.predict_yes(dict.fromkeys(FEATURE_NAMES, 0.0), as_of=datetime(2026, 11, 1, 11, tzinfo=timezone.utc))
        model.predict_proba.assert_not_called()

    def test_disabled_notifications_never_send(self) -> None:
        session = Mock(spec=requests.Session)
        with Notifier(Settings(), session=session) as notifier:
            self.assertFalse(notifier.submit("HALTED", {"reason": "test"}))
        session.post.assert_not_called()

    def test_alert_summaries_whitelist_fields_and_suppress_mentions(self) -> None:
        settings = Settings(alerts_enabled=True, telegram_bot_token=SecretStr("123:test_token"), telegram_chat_id="42",
                            discord_webhook_url=SecretStr("https://discord.com/api/webhooks/123/test_token"))
        session = Mock(spec=requests.Session)
        session.post.return_value = http_response({"ok": True})
        with Notifier(settings, session=session) as notifier:
            notifier.submit("ENTRY_FILL", {"ticker": "KXBTC15M-TEST", "quantity": "1", "api_key": "MUST_NOT_LEAK"})
            notifier._queue.join()
        self.assertEqual(session.post.call_count, 2)
        for call in session.post.call_args_list:
            self.assertNotIn("MUST_NOT_LEAK", json.dumps(call.kwargs["json"]))
            self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(session.post.call_args.kwargs["json"]["allowed_mentions"], {"parse": []})

    def test_untrusted_webhook_destination_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Notifier(Settings(alerts_enabled=True, discord_webhook_url=SecretStr("https://example.com/steal")))

    def test_bootstrap_never_overwrites_buyer_environment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env.example").write_text("KALSHI_TRADING_MODE=paper\n")
            initialize(root)
            self.assertEqual((root / ".env").read_text(), "KALSHI_TRADING_MODE=paper\n")
            self.assertEqual((root / ".env").stat().st_mode & 0o777, 0o600)
            (root / ".env").write_text("BUYER_SETTING=preserve\n")
            initialize(root)
            self.assertEqual((root / ".env").read_text(), "BUYER_SETTING=preserve\n")

    def test_collateral_dry_run_never_creates_a_client(self) -> None:
        with patch("sys.argv", ["transfer_collateral.py", "--amount", "100.00"]), \
                patch.object(collateral, "KalshiClient") as client, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(collateral.main(), 0)
            self.assertIn('"amount_centicents": 1000000', output.getvalue())
            client.from_settings.assert_not_called()

    def test_collateral_exact_units_and_no_transport_retry(self) -> None:
        session = requests.Session()
        self.addCleanup(session.close)
        session.request = Mock(return_value=http_response({"transfer_id": "server-transfer-id"}))
        client = KalshiClient(environment=KalshiEnvironment.PROD, trading_mode=TradingMode.LIVE,
                              live_trading_enabled=True, key_id="test-key", private_key=ed25519.Ed25519PrivateKey.generate(),
                              session=session, sleep=lambda _: None)
        identifier = str(uuid.uuid4())
        self.assertEqual(client.transfer_collateral(Decimal("1.2345"), client_transfer_id=identifier), "server-transfer-id")
        payload = session.request.call_args.kwargs["json"]
        self.assertEqual(payload["amount"], 12345)
        self.assertEqual(payload["source_exchange_shard"], 0)
        self.assertEqual(payload["destination_exchange_shard"], 2)
        self.assertEqual(payload["client_transfer_id"], identifier)
        session.request.reset_mock()
        session.request.side_effect = requests.Timeout()
        with self.assertRaises(OrderSubmissionUnknown):
            client.transfer_collateral(Decimal("1"), client_transfer_id=identifier)
        self.assertEqual(session.request.call_count, 1)

    def test_collateral_invalid_precision_makes_no_request(self) -> None:
        session = requests.Session()
        self.addCleanup(session.close)
        session.request = Mock()
        client = KalshiClient(environment=KalshiEnvironment.PROD, trading_mode=TradingMode.LIVE,
                              live_trading_enabled=True, key_id="test-key", private_key=ed25519.Ed25519PrivateKey.generate(), session=session)
        with self.assertRaises(ValueError):
            client.transfer_collateral(Decimal("0.00001"), client_transfer_id=str(uuid.uuid4()))
        session.request.assert_not_called()

    def test_read_only_performance_groups_partial_exits(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            store = StateStore(path, "test")
            try:
                state = RuntimeState(Phase.FLAT, Decimal(1000))
                now = datetime.now(timezone.utc)
                store.save(state, "ENTRY_FILL", {"ticker": "TEST", "side": "YES", "quantity": "1", "fees": ".02"}, now)
                store.save(state, "EXIT_FILL", {"ticker": "TEST", "pnl_delta": ".10", "remaining": ".5", "fees": ".01", "reason": "TP"}, now)
                store.save(state, "EXIT_FILL", {"ticker": "TEST", "pnl_delta": "-.05", "remaining": "0", "fees": ".01", "reason": "Delta-SL"}, now)
                store.save(state, "SETTLEMENT", {"ticker": "UNTRADED", "pnl_delta": "0"}, now)
                summary = read_performance(path)
                self.assertEqual(summary["trades"], 1)
                self.assertEqual(summary["realized_pnl_usd"], Decimal(".05"))
                self.assertEqual(summary["fees_usd"], Decimal(".04"))
                self.assertEqual(summary["max_realized_drawdown_usd"], Decimal(".05"))
                self.assertEqual(summary["win_percent"], 100)
            finally:
                store.close()
            missing = Path(directory) / "missing.db"
            with self.assertRaises(ValueError):
                read_performance(missing)
            self.assertFalse(missing.exists())

    def test_release_is_deterministic_and_excludes_runtime_secrets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ROOT_FILES:
                (root / name).write_text("")
            for directory_name, names in (("models", MODEL_FILES), ("deploy", DEPLOY_FILES)):
                folder = root / directory_name
                folder.mkdir()
                for name in names:
                    (folder / name).write_bytes(b"fixture")
            model = root / "models" / MODEL_FILES[0]
            (root / "models/manifest.json").write_text(json.dumps({"model": model.name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}))
            (root / ".env").write_text("NEVER_PACKAGE_REAL_KEYS")
            (root / "secret.pem").write_text("NEVER_PACKAGE_REAL_KEYS")
            (root / "state.sqlite3").write_text("NEVER_PACKAGE_REAL_KEYS")
            first = build_release(root, root / "one.zip")
            second = build_release(root, root / "two.zip")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertNotIn("kalshi-ml/.env", archive.namelist())
                self.assertFalse(any(name.endswith((".pem", ".sqlite3")) for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
