"""CLI mode guards and bounded execution without network/account access."""

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main
from kalshi_ml.config import KalshiEnvironment, Settings


class MainTests(unittest.TestCase):
    """Verify paper always uses the production book and live requires opt-in."""

    def test_live_requires_explicit_production_configuration(self) -> None:
        with patch("sys.argv", ["main.py", "--mode", "live"]), patch("sys.stderr", io.StringIO()), \
                patch.object(main, "load_settings", return_value=Settings()), \
                patch.object(main, "KalshiClient") as client, patch.object(main, "load_artifact") as loader:
            self.assertEqual(main.main(), 2)
            client.assert_not_called()
            loader.assert_not_called()

    def test_paper_once_uses_production_without_credentials(self) -> None:
        with TemporaryDirectory() as directory:
            args = ["main.py", "--mode", "paper", "--once", "--state", str(Path(directory) / "state.db"),
                    "--log-file", str(Path(directory) / "events.log")]
            supervisor = Mock()
            supervisor.state = SimpleNamespace(halt_reason=None)
            with patch("sys.argv", args), patch.object(main, "load_settings", return_value=Settings()), \
                    patch.object(main, "load_artifact", return_value={}), patch.object(main, "KalshiClient") as client, \
                    patch.object(main, "MarketDataCollector"), patch.object(main, "TradingDaemon", return_value=supervisor):
                self.assertEqual(main.main(), 0)
                self.assertEqual(client.call_args.kwargs["environment"], KalshiEnvironment.PROD)
                client.from_settings.assert_not_called()
                supervisor.run.assert_called_once_with(iterations=1)

    def test_nonfinite_paper_cash_is_rejected_before_io(self) -> None:
        with patch("sys.argv", ["main.py", "--paper-cash", "NaN"]), patch("sys.stderr", io.StringIO()), \
                patch.object(main, "load_settings") as loader:
            with self.assertRaises(SystemExit) as raised:
                main.main()
            self.assertEqual(raised.exception.code, 2)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
