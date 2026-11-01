"""Preview or explicitly submit a journaled USD transfer from shard 0 to 2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import tempfile
import uuid

from kalshi_ml.config import KalshiEnvironment, TradingMode, load_settings
from kalshi_ml.execution.kalshi_client import KalshiClient, KalshiError


def main() -> int:
    """Dry-run by default; never automatically repeat an unknown transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amount", required=True, help="USD amount, up to four decimal places")
    parser.add_argument("--request-id", help="stable UUID; required for execution")
    parser.add_argument("--execute", action="store_true", help="submit the displayed financial transfer once")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--receipts", type=Path, default=Path("logs/transfers"))
    args = parser.parse_args()
    try:
        amount = Decimal(args.amount)
        if not amount.is_finite() or amount <= 0 or amount * 10000 != (amount * 10000).to_integral_value() or amount * 10000 > 2**63 - 1:
            raise ValueError("amount must be positive USD with at most four decimals")
        request_id = str(uuid.UUID(args.request_id)) if args.request_id else str(uuid.uuid4())
        preview = {"client_transfer_id": request_id, "amount_usd": str(amount), "amount_centicents": int(amount * 10000),
                   "source_exchange_shard": 0, "destination_exchange_shard": 2, "subaccount": 0}
        print(json.dumps(preview, indent=2))
        if not args.execute:
            print("DRY RUN: no account request or transfer sent. Reuse this --request-id with --execute to submit.")
            return 0
        if not args.request_id:
            raise ValueError("--execute requires the previewed --request-id")
        settings = load_settings(args.env_file)
        if settings.environment is not KalshiEnvironment.PROD or settings.runtime.mode is not TradingMode.LIVE or not settings.runtime.live_trading_enabled:
            raise ValueError("execution requires PROD/live/explicit live enablement")
        args.receipts.mkdir(parents=True, exist_ok=True)
        path = args.receipts / f"{request_id}.json"
        # The exclusive intent file prevents this script from resubmitting the ID.
        # A crash or ambiguous response leaves it intact for manual reconciliation.
        with KalshiClient.from_settings(settings) as client:
            if path.exists():
                print("Receipt/intent already exists. Inspect it and transfer history; no new request sent.")
                return 2
            if client.get_balance(exchange_index=0).balance_dollars < amount:
                raise ValueError("insufficient available funds on source shard 0")
            record = {**preview, "status": "intent", "created_at": datetime.now(timezone.utc).isoformat()}
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as output:
                json.dump(record, output, indent=2)
                output.flush()
                os.fsync(output.fileno())
            transfer_id = client.transfer_collateral(amount, client_transfer_id=request_id)
            record.update(status="accepted_async", transfer_id=transfer_id)
            descriptor, temporary = tempfile.mkstemp(dir=args.receipts, prefix=".receipt-")
            try:
                with os.fdopen(descriptor, "w") as output:
                    json.dump(record, output, indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            print(f"Accepted asynchronously: transfer_id={transfer_id}. Verify transfer history/destination balance before trading.")
        return 0
    except (ValueError, OSError, ArithmeticError, KalshiError) as exc:
        # Never print requests' exception URLs, signatures or private keys.
        if isinstance(exc, KalshiError) and getattr(exc, "client_order_id", None):
            print(f"Transfer unresolved/rejected; reconcile request ID {exc.client_order_id}. No automatic retry.")
        else:
            print(f"Transfer not completed: {type(exc).__name__}; inspect configuration/receipt and account history.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
