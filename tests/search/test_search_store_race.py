"""Store: creating the same new index from several processes / threads at once."""

from __future__ import annotations

import multiprocessing as mp
import queue
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from facesorter.search.errors import IndexMismatchError
from facesorter.search.store import Store
from facesorter.search.types import PhotoMeta

WAIT_S = 30  # every wait in the process tests; a hang fails fast instead of stalling the suite


def _open_and_report(base, dim, rounds, barrier, results):
    """Child-process body (top level so it pickles under ``spawn``).

    For each round: line up on the shared barrier, open a fresh path at the same instant as
    the other children, and report ``(round, outcome, dim)``.
    """
    for round_ in range(rounds):
        try:
            barrier.wait(timeout=WAIT_S)
        except Exception:  # noqa: BLE001 - a broken barrier: let the parent time out
            return
        try:
            with Store.open(Path(base) / f"r{round_}" / "i.db", model_name="m", dim=dim):
                results.put((round_, "ok", dim))
        except Exception as exc:  # noqa: BLE001 - the type name is the result
            results.put((round_, type(exc).__name__, dim))


def _race_processes(base, dims, rounds):
    """Race ``len(dims)`` spawned processes on ``rounds`` fresh index paths.

    ``spawn`` (not ``fork``): forking after NumPy / Accelerate threads exist hangs on macOS.
    Returns ``{round: [(outcome, dim), ...]}``; any hang terminates every child and fails.
    """
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(len(dims)), ctx.Queue()
    procs = [
        ctx.Process(target=_open_and_report, args=(str(base), d, rounds, barrier, results))
        for d in dims
    ]
    outcomes: dict[int, list[tuple[str, int]]] = {r: [] for r in range(rounds)}
    try:
        for proc in procs:
            proc.start()
        for _ in range(rounds * len(dims)):
            round_, status, dim = results.get(timeout=WAIT_S)
            outcomes[round_].append((status, dim))
        for proc in procs:
            proc.join(timeout=WAIT_S)
    except queue.Empty:
        pytest.fail(f"racing processes did not report within {WAIT_S}s: {outcomes}")
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
    return outcomes


def test_processes_racing_to_create_the_same_index(tmp_path):
    outcomes = _race_processes(tmp_path, [8, 8, 8, 8], rounds=6)
    for round_, got in outcomes.items():
        assert got == [("ok", 8)] * 4, (round_, got)
        with Store.open(tmp_path / f"r{round_}" / "i.db", create=False) as store:
            assert (store.model_name, store.dim, store.count()) == ("m", 8, 0)  # one valid index


def test_processes_racing_with_different_dims_leave_one_valid_index(tmp_path):
    outcomes = _race_processes(tmp_path, [8, 16, 8, 16], rounds=6)
    for round_, got in outcomes.items():
        path = tmp_path / f"r{round_}" / "i.db"
        winners = {dim for status, dim in got if status == "ok"}
        losers = [(status, dim) for status, dim in got if status != "ok"]
        assert len(winners) == 1, (round_, got)  # exactly one dimension won
        assert losers and all(status == "IndexMismatchError" for status, _ in losers), got
        assert all(dim not in winners for _, dim in losers), got
        assert path.exists()  # a loser never deletes the winner's file
        with Store.open(path, create=False) as store:
            assert store.dim in winners


def test_threads_racing_to_create_the_same_index(tmp_path):
    for round_ in range(10):
        path = tmp_path / f"t{round_}.db"
        barrier = threading.Barrier(6)
        outcomes: list[object] = []

        def work(dim: int, path=path, barrier=barrier, outcomes=outcomes) -> None:
            barrier.wait()
            try:
                with Store.open(path, model_name="m", dim=dim) as store:
                    outcomes.append(store.dim)
            except IndexMismatchError:
                outcomes.append("mismatch")

        threads = [threading.Thread(target=work, args=(8 if i % 3 else 12,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert len(outcomes) == 6
        ok = {o for o in outcomes if o != "mismatch"}
        assert len(ok) == 1 and "mismatch" in outcomes  # both dims are in play, one wins


def test_create_schema_reports_losing_the_race_without_touching_the_index(tmp_path):
    path = tmp_path / "i.db"
    with Store.open(path, model_name="m", dim=8) as store:
        store.upsert_photos(
            [(PhotoMeta(path="/a", size=1, mtime_ns=1), None, np.ones(8, dtype=np.float32))]
        )
    raw = sqlite3.connect(path, isolation_level=None)
    assert Store._create_schema(raw, "other", 99) is False
    assert raw.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone() == ("8",)
    assert raw.execute("SELECT COUNT(*) FROM photos").fetchone() == (1,)
    assert raw.in_transaction is False
    raw.close()


@pytest.mark.parametrize(("dim", "ok"), [(8, True), (9, False)])
def test_loser_validates_against_the_winners_meta(tmp_path, monkeypatch, dim, ok):
    """Simulate: this process saw an empty file, but the index appeared before it got the lock."""
    path = tmp_path / "i.db"
    Store.open(path, model_name="m", dim=8).close()
    real = Store._read_state
    calls = []

    def stale_first_read(db, p):
        calls.append(1)
        return (0, 0, False) if len(calls) == 1 else real(db, p)

    monkeypatch.setattr(Store, "_read_state", staticmethod(stale_first_read))
    if ok:
        with Store.open(path, model_name="m", dim=dim) as store:
            assert store.dim == 8
    else:
        with pytest.raises(IndexMismatchError, match="Re-index"):
            Store.open(path, model_name="m", dim=dim)
    assert path.exists() and len(calls) == 2


def test_unexpected_failure_while_creating_removes_the_empty_file(tmp_path, monkeypatch):
    def boom(db, *args):
        db.execute("PRAGMA user_version").fetchone()  # makes SQLite create the empty file
        raise RuntimeError("boom")

    monkeypatch.setattr(Store, "_initialise", classmethod(lambda cls, db, *a: boom(db, *a)))
    path = tmp_path / "new.db"
    with pytest.raises(RuntimeError, match="boom"):
        Store.open(path, model_name="m", dim=8)
    assert list(tmp_path.iterdir()) == []
