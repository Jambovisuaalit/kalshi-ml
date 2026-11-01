"""Build a deterministic source distribution from an explicit safe allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT_FILES = (".env.example", ".gitignore", ".dockerignore", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
              "requirements.txt", "pyproject.toml", "quickstart.py", "main.py")
MODEL_FILES = ("hgb_btc15m_prod.joblib", "hgb_btc15m_prod.metadata.json", "manifest.json", "MODEL_CARD.md")
DEPLOY_FILES = ("Dockerfile", "docker-compose.yml", "kalshi-bot.service")


def release_files(root: Path) -> list[Path]:
    """Never discover runtime files recursively outside known source patterns."""
    paths = [root / name for name in ROOT_FILES]
    paths.extend(root / "models" / name for name in MODEL_FILES)
    paths.extend(root / "deploy" / name for name in DEPLOY_FILES)
    for folder in ("kalshi_ml", "scripts", "tests"):
        paths.extend((root / folder).rglob("*.py"))
    paths.extend((root / "docs").glob("*.md"))
    paths.extend((root / ".github/workflows").glob("*.yml"))
    for path in paths:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise ValueError("symlinks are not allowed in a release")
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("missing or out-of-root release file")
    for line in (root / ".env.example").read_text().splitlines():
        key, _, value = line.partition("=")
        if key in {"KALSHI_KEY_ID", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DISCORD_WEBHOOK_URL"} and value.strip():
            raise ValueError("release .env.example must not contain credentials/destinations")
    manifest = json.loads((root / "models/manifest.json").read_text())
    if manifest["model"] != MODEL_FILES[0] or hashlib.sha256((root / "models" / MODEL_FILES[0]).read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("bundled model checksum mismatch")
    return sorted(set(paths), key=lambda path: str(path.relative_to(root)))


def build_release(root: Path, destination: Path) -> Path:
    """Write atomically; runtime DBs, keys, logs and raw data are never candidates."""
    root = root.resolve()
    paths = release_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".release-", dir=destination.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                info = zipfile.ZipInfo("kalshi-ml/" + path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest() + "  " + destination.name + "\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/kalshi-ml-0.2.0.zip"))
    args = parser.parse_args()
    path = build_release(Path(__file__).resolve().parents[1], args.output)
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
