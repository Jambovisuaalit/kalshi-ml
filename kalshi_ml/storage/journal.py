"""Durable SQLite journal, execution checkpoints and read-only PnL reporting.

Unix file locking supports macOS/Linux. Checkpoints retain their existing
JSON shape for backward-compatible recovery across this module extraction.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
import fcntl
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
from kalshi_ml.execution.kalshi_client import TargetSide

class Phase(StrEnum):
    """Persisted execution states; a data outage never implies a flat position."""

    WARMING = "WARMING"
    WAITING_MARKET = "WAITING_MARKET"
    FLAT = "FLAT"
    OPENING = "OPENING"
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    SETTLING = "SETTLING"
    HALTED = "HALTED"
    STOPPED = "STOPPED"


def parse_datetime(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("state timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


@dataclass(slots=True)
class CycleLock:
    """Official strike is immutable after first observation in this process."""

    ticker: str
    start: datetime
    end: datetime
    strike: float | None
    strike_available_at: datetime | None
    traded: bool = False
    last_settlement_check: datetime | None = None


@dataclass(slots=True)
class Position:
    """Remaining owned-leg quantity and its unallocated entry basis in dollars."""

    ticker: str
    side: TargetSide
    quantity: Decimal
    average_entry: Decimal
    basis: Decimal
    strike: float
    end: datetime


@dataclass(slots=True)
class PendingOrder:
    """Durable intent; an unknown exchange ID is reconciled through client ID."""

    client_id: str
    ticker: str
    side: TargetSide
    closing: bool
    count: int
    price_cents: int
    reason: str
    exchange_id: str | None = None


@dataclass(slots=True)
class RuntimeState:
    """Single atomic checkpoint paired with the journal's immutable events."""

    phase: Phase
    cash: Decimal
    realized_pnl: Decimal = Decimal(0)
    position: Position | None = None
    cycle: CycleLock | None = None
    unsettled: list[CycleLock] = field(default_factory=list)
    pending: PendingOrder | None = None
    halt_reason: str | None = None


class StateStore:
    """SQLite FULL durability and an OS process lock (macOS/Linux).

    Use one state path per account/mode. Never delete a live journal to bypass
    reconciliation. The lock releases automatically if the process crashes.
    """

    def __init__(self, path: Path, identity: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = path.with_suffix(path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("another daemon owns this state path") from None
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY, identity TEXT NOT NULL, state TEXT NOT NULL)")
        self._connection.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, event TEXT NOT NULL, details TEXT NOT NULL)")
        self._identity = identity
        row = self._connection.execute("SELECT identity FROM checkpoint WHERE id=1").fetchone()
        if row is not None and row[0] != identity:
            self.close()
            raise ValueError("state path belongs to another mode/account/configuration")

    def load(self) -> RuntimeState | None:
        """Reconstruct exact monetary values; malformed checkpoints fail startup."""
        row = self._connection.execute("SELECT state FROM checkpoint WHERE id=1").fetchone()
        if row is None:
            return None
        raw = json.loads(row[0])

        def cycle(value: dict[str, Any]) -> CycleLock:
            return CycleLock(value["ticker"], parse_datetime(value["start"]), parse_datetime(value["end"]), value["strike"],
                             parse_datetime(value["strike_available_at"]) if value["strike_available_at"] else None,
                             value["traded"], parse_datetime(value["last_settlement_check"]) if value["last_settlement_check"] else None)

        position = raw["position"]
        restored = None if position is None else Position(
            position["ticker"], position["side"], Decimal(position["quantity"]), Decimal(position["average_entry"]),
            Decimal(position["basis"]), position["strike"], parse_datetime(position["end"]),
        )
        state = RuntimeState(
            Phase(raw["phase"]), Decimal(raw["cash"]), Decimal(raw["realized_pnl"]), restored,
            cycle(raw["cycle"]) if raw["cycle"] else None, [cycle(value) for value in raw["unsettled"]],
            PendingOrder(**raw["pending"]) if raw["pending"] else None, raw["halt_reason"],
        )
        if not state.cash.is_finite() or not state.realized_pnl.is_finite():
            raise ValueError("invalid journal monetary values")
        if restored and (
            restored.side not in {"YES", "NO"} or not restored.quantity.is_finite() or restored.quantity <= 0
            or not restored.basis.is_finite() or restored.basis < 0
            or not restored.average_entry.is_finite() or not 0 <= restored.average_entry <= 1
            or not math.isfinite(restored.strike) or restored.strike <= 0
        ):
            raise ValueError("invalid journal position")
        if state.pending and (state.pending.side not in {"YES", "NO"} or type(state.pending.closing) is not bool
                              or type(state.pending.count) is not int or state.pending.count < 1
                              or not 1 <= state.pending.price_cents <= 99):
            raise ValueError("invalid journal order intent")
        return state

    def save(self, state: RuntimeState, event: str, details: dict[str, Any], now: datetime) -> None:
        """Commit state and event together, including before any order mutation."""
        encoded = json.dumps(asdict(state), default=str, allow_nan=False)
        with self._connection:
            self._connection.execute("INSERT OR REPLACE INTO checkpoint VALUES (1, ?, ?)", (self._identity, encoded))
            self._connection.execute("INSERT INTO events(timestamp,event,details) VALUES (?,?,?)",
                                     (now.isoformat(), event, json.dumps(details, default=str, allow_nan=False)))

    def close(self) -> None:
        """Release the database and process lock; caller must stop polling first."""
        self._connection.close()
        self._lock.close()


def read_performance(path: Path) -> dict[str, Any]:
    """Read an existing journal without a write lock or schema modification.

    Drawdown is REALIZED-equity drawdown; historical quote snapshots are not
    stored, so a mark-to-market drawdown cannot be reconstructed from this DB.
    A trade finishes only when its remaining quantity reaches zero or settles.
    """
    from urllib.parse import quote

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError("journal does not exist")
    connection = sqlite3.connect("file:" + quote(str(path), safe="/") + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute("SELECT timestamp,event,details FROM events ORDER BY id").fetchall()
        checkpoint = connection.execute("SELECT state FROM checkpoint WHERE id=1").fetchone()
    finally:
        connection.close()
    opened: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    realized = fees = peak = drawdown = Decimal(0)
    unmatched = 0
    for timestamp, event, encoded in rows:
        details = json.loads(encoded)
        ticker = details.get("ticker")
        if event == "ENTRY_FILL" and Decimal(str(details.get("quantity", "0"))) > 0:
            opened[ticker] = {"ticker": ticker, "side": details.get("side"), "entry_ts": timestamp,
                              "quantity": details["quantity"], "net_pnl": Decimal(0)}
        if event in {"ENTRY_FILL", "EXIT_FILL"}:
            fees += Decimal(str(details.get("fees", "0")))
        if event not in {"EXIT_FILL", "SETTLEMENT"}:
            continue
        delta = Decimal(str(details.get("pnl_delta", "0")))
        realized += delta
        peak = max(peak, realized)
        drawdown = max(drawdown, peak - realized)
        trade = opened.get(ticker)
        if trade is None:
            if event == "EXIT_FILL":
                unmatched += 1
            continue
        trade["net_pnl"] += delta
        terminal = event == "SETTLEMENT" or Decimal(str(details.get("remaining", "0"))) == 0
        if terminal:
            trade.update(exit_ts=timestamp, exit_reason="Settlement" if event == "SETTLEMENT" else details.get("reason"))
            trades.append(trade)
            del opened[ticker]
    wins = sum(trade["net_pnl"] > 0 for trade in trades)
    profits = sum((max(Decimal(0), trade["net_pnl"]) for trade in trades), Decimal(0))
    losses = -sum((min(Decimal(0), trade["net_pnl"]) for trade in trades), Decimal(0))
    factor: Decimal | str | None = profits / losses if losses else ("inf" if profits else None)
    snapshot = json.loads(checkpoint[0]) if checkpoint else {}
    return {"trades": len(trades), "win_percent": 100 * wins / len(trades) if trades else 0.0,
            "realized_pnl_usd": realized, "fees_usd": fees, "profit_factor": factor,
            "max_realized_drawdown_usd": drawdown, "open_trades": len(opened),
            "unmatched_closing_events": unmatched, "cash_usd": snapshot.get("cash"),
            "phase": snapshot.get("phase"), "trade_records": trades}


def performance_main() -> int:
    """Print read-only journal statistics and optionally export closed trades."""
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("logs/paper-exchange-2.sqlite3"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", type=Path, help="optional closed-trade export destination")
    args = parser.parse_args()
    try:
        summary = read_performance(args.state)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=["ticker", "side", "entry_ts", "quantity", "net_pnl", "exit_ts", "exit_reason"])
                writer.writeheader()
                writer.writerows(summary["trade_records"])
        if args.json:
            print(json.dumps(summary, default=str, indent=2))
        else:
            print(f"Closed trades: {summary['trades']} | Win: {summary['win_percent']:.2f}%")
            print(f"Realized net P&L: ${summary['realized_pnl_usd']:.4f} | Fees: ${summary['fees_usd']:.4f}")
            factor = "N/A" if summary["profit_factor"] is None else summary["profit_factor"]
            print(f"Profit Factor: {factor} | Realized-equity drawdown: ${summary['max_realized_drawdown_usd']:.4f}")
            print(f"Open trades: {summary['open_trades']} | Phase: {summary['phase']}")
            if summary["unmatched_closing_events"]:
                print("Incomplete history: unmatched closing events; trade-level statistics may be incomplete.")
        return 0
    except (ValueError, KeyError, OSError, ArithmeticError, sqlite3.Error):
        print("Cannot read performance: inspect the journal path/schema; no database changes were made.")
        return 2

