"""One-command initialization and read-only local system checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def _resource_root(root: Path) -> Path:
    if (root / ".env.example").is_file():
        return root
    installed = Path(sys.prefix) / "share/kalshi-ml"
    if (installed / ".env.example").is_file():
        return installed
    raise ValueError("cannot locate bundled .env.example; run from the extracted source directory")


def initialize(root: Path) -> None:
    """Create directories/config without replacing credentials or trained models."""
    source = _resource_root(root)
    for name in ("logs", "models", "secrets", "data/raw", "data/processed"):
        (root / name).mkdir(parents=True, exist_ok=True)
    destination = root / ".env"
    if not destination.exists():
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as output:
                output.write((source / ".env.example").read_bytes())
    if source != root:
        for filename in ("hgb_btc15m_prod.joblib", "hgb_btc15m_prod.metadata.json", "manifest.json", "MODEL_CARD.md"):
            target = root / "models" / filename
            if not target.exists():
                shutil.copy2(source / "models" / filename, target)


def check_system(root: Path, *, network: bool = False) -> None:
    """Check dependency pins, Pydantic config and trusted bundled model; no orders."""
    from kalshi_ml.config import FEATURE_NAMES, KalshiEnvironment, TradingMode, load_settings
    from kalshi_ml.execution.kalshi_client import ClientPolicy, KalshiClient, load_signing_key
    from kalshi_ml.model.inference import RealtimePredictor

    source = _resource_root(root)
    for line in (source / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"dependency differs from release lock: {name}")
    settings = load_settings(root / ".env")
    print(f"OK: Python {sys.version.split()[0]}, exact dependencies, configuration, shard {settings.runtime.exchange_index}")
    path = root / "models/hgb_btc15m_prod.joblib"
    manifest = json.loads((root / "models/manifest.json").read_text())
    if manifest["model"] != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("bundled model checksum mismatch")
    predictor = RealtimePredictor.from_file(path)
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    features.update(remaining_sec=600.0, vol_15m=0.01, vol_60m=0.02, vol_ratio=0.5)
    probability = predictor.predict_yes(features)
    print(f"OK: model checksum/schema/test-vector inference; P(YES)={probability:.4f}; provenance={predictor.metadata.get('price_mode')}")
    if settings.runtime.mode is TradingMode.LIVE:
        settings.validate_credentials()
        load_signing_key(settings.private_key_path)
        print("OK: local live credential/key validation (exchange authentication not tested)")
    if settings.alerts_enabled:
        from kalshi_ml.monitoring.notifier import Notifier

        # Validate destinations without starting the background sender.
        notifier = Notifier(settings)
        notifier.close()
        print("OK: notification destinations validated; no messages sent")
    if network:
        with KalshiClient(environment=KalshiEnvironment.PROD, exchange_index=settings.runtime.exchange_index,
                          policy=ClientPolicy(timeout_sec=3, max_read_retries=0)) as client:
            market = client.get_active_15m_market()
            print("OK: public market endpoint; " + (market.ticker if market else "no active market in the configured cycle/shard"))
    print("No exchange order, collateral transfer or notification was sent.")


def main() -> int:
    """Default source setup bootstraps an isolated venv; --check never installs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="read-only local checks; do not install or initialize")
    parser.add_argument("--network", action="store_true", help="also read the public production market endpoint")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required. Run this script with python3.11 or python3.12.")
        return 2
    root = args.root.expanduser().resolve()
    try:
        if not args.check:
            initialize(root)
            if (root / "pyproject.toml").is_file():
                target = root / ".venv"
                if not (target / "bin/python").exists():
                    venv.EnvBuilder(with_pip=True).create(target)
                python = target / "bin/python"
                subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=True, cwd=root)
                subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(root)], check=True, cwd=root)
                command = [str(python), "-m", "kalshi_ml.setup", "--check", "--root", str(root)]
                if args.network:
                    command.append("--network")
                return subprocess.run(command, cwd=root, check=False).returncode
        check_system(root, network=args.network)
        print("Start paper mode: kalshi-ml-run --mode paper (or .venv/bin/python main.py --mode paper)")
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, KeyError, importlib.metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Setup/check failed: {type(exc).__name__}. Check Python, dependency lock, .env and model files; secrets are not printed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
