"""Start the supervisor: python main.py --mode paper (default) or --mode live."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
import logging
from pathlib import Path
import signal
import sqlite3
import sys
import time

from kalshi_ml.config import KalshiEnvironment, TradingMode, load_settings
from kalshi_ml.daemon import DaemonConfig, StateStore, StatusConsole, TradingDaemon, state_identity
from kalshi_ml.data.collector import MarketDataCollector, RetryPolicy
from kalshi_ml.execution.kalshi_client import ClientPolicy, KalshiClient
from kalshi_ml.model.trainer import DEFAULT_MODEL_PATH, load_artifact
from kalshi_ml.model.inference import default_model_path
from kalshi_ml.monitoring.notifier import Notifier


class ConsoleLogHandler(logging.StreamHandler):
    """Clear the status line before printing complete timestamped log records."""

    def __init__(self, console: StatusConsole) -> None:
        super().__init__(sys.stdout)
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        self.console.clear()
        super().emit(record)


def _money(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("expected a finite monetary amount") from None
    if not amount.is_finite() or amount < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative monetary amount")
    return amount


def main() -> int:
    """Validate configuration, lock the journal, install stop handlers and run.

    This entry point never upgrades paper credentials/mode implicitly. Live
    execution additionally requires the existing PROD/live/enable settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="trusted local joblib artifact")
    parser.add_argument("--state", type=Path, help="persistent SQLite journal; default logs/<mode>.sqlite3")
    parser.add_argument("--log-file", type=Path, help="default logs/<mode>.log")
    parser.add_argument("--paper-cash", type=_money, default=Decimal("1000.00"))
    parser.add_argument("--paper-fee", type=_money, default=Decimal("0.02"))
    parser.add_argument("--contracts", type=int, default=1)
    bounded = parser.add_mutually_exclusive_group()
    bounded.add_argument("--once", action="store_true", help="run exactly one poll")
    bounded.add_argument("--iterations", type=int, help="stop after N polls")
    args = parser.parse_args()
    mode = TradingMode(args.mode)
    console = StatusConsole()
    root = logging.getLogger()
    previous_level = root.level
    handlers: list[logging.Handler] = []
    old_signals: dict[int, object] = {}
    try:
        settings = load_settings(args.env_file)
        config = DaemonConfig(mode=mode, contracts=args.contracts, initial_paper_cash=args.paper_cash,
                              paper_fee_per_contract=args.paper_fee)
        if args.iterations is not None and args.iterations < 1:
            raise ValueError("--iterations must be positive")
        if mode is TradingMode.LIVE and (
            settings.environment is not KalshiEnvironment.PROD or settings.runtime.mode is not TradingMode.LIVE
            or not settings.runtime.live_trading_enabled
        ):
            raise ValueError("live mode requires KALSHI_ENV=PROD, KALSHI_TRADING_MODE=live and KALSHI_LIVE_TRADING_ENABLED=true")
        artifact = load_artifact(args.model)
        log_path = args.log_file or Path(f"logs/{mode.value}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [ConsoleLogHandler(console), logging.FileHandler(log_path, encoding="utf-8")]
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        formatter.converter = time.gmtime
        for handler in handlers:
            handler.setFormatter(formatter)
            root.addHandler(handler)
        root.setLevel(logging.INFO)
        # The outer four-second loop retries reads. Do not block risk polling
        # behind the reusable client's multi-attempt historical-read policy.
        timeout = min(float(settings.http_timeout_sec), 3.0)
        policy = ClientPolicy(timeout_sec=timeout, max_read_retries=0)
        with ExitStack() as resources:
            client = resources.enter_context(
                KalshiClient.from_settings(settings, policy=policy) if mode is TradingMode.LIVE
                else KalshiClient(environment=KalshiEnvironment.PROD, policy=policy, exchange_index=settings.runtime.exchange_index)
            )
            collector = resources.enter_context(MarketDataCollector(retry=RetryPolicy(
                max_retries=0, timeout_sec=timeout,
            )))
            store = StateStore(args.state or Path(f"logs/{mode.value}-exchange-{settings.runtime.exchange_index}.sqlite3"),
                               state_identity(mode, settings.key_id, config.contracts, config.initial_paper_cash, config.paper_fee_per_contract,
                                              exchange_index=settings.runtime.exchange_index))
            resources.callback(store.close)
            notifier = resources.enter_context(Notifier(settings))
            daemon = TradingDaemon(client, collector, artifact, store, config=config, strategy=settings.strategy,
                                   console=console, notifier=notifier)

            def request_stop(signum: int, frame: object) -> None:
                daemon.stop_event.set()

            for signum in (signal.SIGINT, signal.SIGTERM):
                old_signals[signum] = signal.signal(signum, request_stop)
            daemon.run(iterations=1 if args.once else args.iterations)
            return 2 if daemon.state.halt_reason else 0
    except (ValueError, OSError, RuntimeError, ArithmeticError, sqlite3.Error) as exc:
        console.clear()
        print(f"Startup/runtime error: {exc}", file=sys.stderr)
        return 2
    finally:
        for signum, previous in old_signals.items():
            signal.signal(signum, previous)
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous_level)


if __name__ == "__main__":
    raise SystemExit(main())
