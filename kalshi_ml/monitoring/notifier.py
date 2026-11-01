"""Bounded Telegram/Discord notifications outside the trading thread.

Disabled by default. Enabling alerts and specifying a destination explicitly
opts the running program into sending trading-event summaries there. Secrets
and arbitrary event fields are never forwarded. Failed sends are not retried.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import queue
import re
import threading
from typing import Any
from urllib.parse import urlsplit

import requests

from kalshi_ml.config import Settings

LOGGER = logging.getLogger(__name__)
ALERT_EVENTS = frozenset({"START", "ENTRY_FILL", "EXIT_FILL", "SETTLEMENT", "HALTED", "ORDER_UNKNOWN", "SHUTDOWN_PENDING", "STOP"})
PUBLIC_FIELDS = frozenset({"ticker", "side", "quantity", "average_fill_price", "pnl_delta", "realized_pnl", "cash", "reason", "mode", "exchange_index"})


class Notifier:
    """At-most-once delivery per configured channel, with bounded shutdown."""

    def __init__(self, settings: Settings, *, session: requests.Session | None = None, capacity: int = 100) -> None:
        if capacity < 1:
            raise ValueError("notification capacity must be positive")
        self.enabled = settings.alerts_enabled
        self._telegram_token = settings.telegram_bot_token.get_secret_value()
        self._chat_id = settings.telegram_chat_id
        self._discord_url = settings.discord_webhook_url.get_secret_value()
        if self.enabled:
            if bool(self._telegram_token) != bool(self._chat_id):
                raise ValueError("Telegram requires both token and chat ID")
            if self._telegram_token and not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", self._telegram_token):
                raise ValueError("invalid Telegram token format")
            if self._discord_url:
                try:
                    parsed = urlsplit(self._discord_url)
                    valid = (parsed.scheme == "https" and parsed.hostname == "discord.com" and not parsed.username and not parsed.password
                             and parsed.port in (None, 443) and not parsed.query and not parsed.fragment
                             and re.fullmatch(r"/api(?:/v[0-9]+)?/webhooks/[0-9]+/[A-Za-z0-9._-]+", parsed.path))
                except ValueError:
                    valid = False
                if not valid:
                    raise ValueError("Discord webhook must be a direct HTTPS discord.com webhook URL")
            if not self._telegram_token and not self._discord_url:
                raise ValueError("enabled alerts require a Telegram or Discord destination")
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        if self._owns_session:
            self._session.trust_env = False
        self._queue: queue.Queue[str] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def __enter__(self) -> Notifier:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        """Start one background sender; disabled instances do no network work."""
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="kalshi-alerts", daemon=True)
            self._thread.start()

    def submit(self, event: str, details: Mapping[str, Any]) -> bool:
        """Queue only approved summaries; never block a risk check on HTTP."""
        if not self.enabled or event not in ALERT_EVENTS or self._stop.is_set():
            return False
        public = {key: value for key, value in details.items() if key in PUBLIC_FIELDS}
        text = (f"kalshi-ml | {event}\n" + json.dumps(public, default=str, ensure_ascii=False, sort_keys=True))[:1800]
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            self.dropped += 1
            LOGGER.warning("Alert queue full; event dropped")
            return False

    def _send(self, url: str, payload: dict[str, Any], *, telegram: bool) -> None:
        try:
            with self._session.post(url, json=payload, timeout=(2.0, 3.0), allow_redirects=False) as response:
                if response.status_code not in (200, 204):
                    LOGGER.warning("Alert delivery rejected (HTTP %s); not retried", response.status_code)
                elif telegram and response.json().get("ok") is not True:
                    LOGGER.warning("Telegram did not acknowledge the alert; not retried")
        except (requests.RequestException, ValueError, AttributeError):
            # Exception messages/prepared request URLs can contain the token.
            LOGGER.warning("Alert delivery failed; request details redacted; not retried")

    def _worker(self) -> None:
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    text = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if self._telegram_token:
                        self._send(f"https://api.telegram.org/bot{self._telegram_token}/sendMessage",
                                   {"chat_id": self._chat_id, "text": text}, telegram=True)
                    if self._discord_url:
                        self._send(self._discord_url, {"content": text, "allowed_mentions": {"parse": []}}, telegram=False)
                finally:
                    self._queue.task_done()
        finally:
            if self._owns_session:
                self._session.close()

    def close(self) -> None:
        """Give the background sender at most two seconds to drain on shutdown."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        elif self._owns_session:
            self._session.close()
