"""Read-only historical fills and realized-PnL reporting."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kalshi_ml.storage.journal import performance_main

if __name__ == "__main__":
    raise SystemExit(performance_main())
