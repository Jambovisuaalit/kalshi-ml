# Third-party components

The proprietary license applies to this project's original files only.
Dependencies are installed separately from `requirements.txt`; their licenses
remain in their distributions and are not replaced by LICENSE.

| Component | Upstream license family |
| --- | --- |
| NumPy, pandas, SciPy, scikit-learn, joblib | BSD and bundled component notices |
| requests, python-dateutil | Apache/BSD as stated upstream |
| cryptography | Apache-2.0 or BSD-3-Clause; bundled notices also apply |
| Pydantic, python-dotenv | MIT |

The full exact dependency list is in requirements.txt. Installed `.dist-info`
license files are authoritative for each installed version, including all
transitive/bundled components. Container builds retain those files.
Kalshi/Coinbase/Kraken API access and market-data use remain governed by their
respective service terms. Telegram and Discord are optional external services.
No third-party affiliation or endorsement is implied.
