FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --only-binary=:all: -r requirements.txt

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY .env.example ./
COPY kalshi_ml/ ./kalshi_ml/
COPY models/hgb_btc15m_prod.joblib models/hgb_btc15m_prod.metadata.json models/manifest.json models/MODEL_CARD.md ./models/
RUN python -m pip install --no-deps . && \
    groupadd --gid 10001 kalshi && \
    useradd --uid 10001 --gid 10001 --create-home kalshi && \
    mkdir -p /app/logs /app/secrets && \
    chown kalshi:kalshi /app/logs /app/secrets

USER 10001:10001
STOPSIGNAL SIGTERM
CMD ["kalshi-ml-run", "--mode", "paper"]
