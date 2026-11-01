"""Local configuration inspection commands; no trading commands yet."""

import argparse
import json
from dataclasses import asdict
from decimal import InvalidOperation

from kalshi_ml.config import FEATURE_NAMES, SERIES_TICKER, load_settings


def main() -> int:
    """Validate settings and optionally credentials without exposing secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-config", "show-config"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--check-credentials", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings(args.env_file)
        if args.check_credentials:
            settings.validate_credentials()
    except (ValueError, TypeError, OSError, InvalidOperation):
        # Parser exceptions may contain raw environment values; keep secrets out.
        parser.exit(2, "Configuration invalid; check environment, thresholds and credential file.\n")
    if args.command == "show-config":
        public = {
            "series": SERIES_TICKER,
            "environment": settings.environment,
            "runtime": asdict(settings.runtime),
            "strategy": asdict(settings.strategy),
            "features": FEATURE_NAMES,
            "market_poll_sec": settings.market_poll_sec,
            "position_poll_sec": settings.position_poll_sec,
            "http_timeout_sec": settings.http_timeout_sec,
            "http_max_retries": settings.http_max_retries,
        }
        print(json.dumps(public, default=str, indent=2))
    else:
        print("Configuration valid.")
    return 0
