"""Source-checkout daemon entry point; also installed as kalshi-ml-run."""
import sys
from kalshi_ml import runner
if __name__ == "__main__":
    raise SystemExit(runner.main())
else:
    # Compatibility for existing embedders/tests importing the original main module.
    sys.modules[__name__] = runner
