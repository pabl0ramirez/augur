"""
Augur cost test: does the budgeted campaign reach a fixed accuracy at lower cost than running every
circuit once at the top setting? The protocol is benchmarks/COST_TEST_PREREG.md; this file runs it.

    python benchmarks/cost_test.py seal              # may the sealed run start?
    python benchmarks/cost_test.py dev               # tiny end-to-end run, writes nothing
    python benchmarks/cost_test.py run               # the sealed run, once (hours)
    python benchmarks/cost_test.py verify RESULT     # re-apply the target and the rule
    python benchmarks/cost_test.py verify RESULT --reference   # and recompute the exact answers

Every engine run happens in a child process, so a cut at the cap is a real kill and one run's
memory never carries into the next. `verify` recomputes, from the values and costs recorded in the
result, which circuits reached the target and how the rule decides; it cannot re-time the runs,
which is why the protocol takes three repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from augur import core as A

HERE = Path(__file__).resolve().parent
PREREG = HERE / "COST_TEST_PREREG.md"
PREREG_SHA256 = "70612a82d0c7b5d393f494cb67521ba1d94f7aa528dff95e393868a72206231d"
OTS = PREREG.with_name(PREREG.name + ".ots")
RESULTS = HERE / "results"
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"

# ── the sealed protocol (prereg sections 2 to 6) ─────────────────────────────────────────────────
SEALED: Dict[str, Any] = {
    "tiers": {"S": [3, 6], "L": [4, 6]},
    "ki_steps": [3, 5, 8], "ki_thetas": [0.3, 0.7, 1.0], "bw_steps": [3, 5, 8],
    "naive_chi": 2048, "naive_cap_s": 600.0,
    "budget_per_circuit_s": 300.0, "augur_kill_s": 7800.0,
    "threads": "4", "repeats": 3, "target_r2": 0.99, "flat_tol": 0.01,
}
#: A child gets this much address space (RLIMIT_AS); a run that needs more fails instead of growing
#: past it. POSIX only: the harness is run on Linux.
CHILD_MEMORY_BYTES = 14 * 10**9
#: The exact reference for one tier must finish within this.
REFERENCE_TIMEOUT_S = 3600.0
#: The child prints its result on one line after this marker; nothing else is parsed.
SENTINEL = "COST_TEST_RESULT "


# ── circuits, target, rule ───────────────────────────────────────────────────────────────────────

def circuits(cfg: Dict[str, Any], tier: str) -> List[Tuple[str, Any, List[Any]]]:
    """The `bench` grid at one tier's size: (id, circuit, parsed observables), in a fixed order."""
    rows, cols = cfg["tiers"][tier]
    n = rows * cols
    edges = A.heavy_hex_edges(rows, cols)
    chain = [(q, q + 1) for q in range(n - 1)]
    out = []
    for t in cfg["ki_steps"]:
        for th in cfg["ki_thetas"]:
            out.append((f"ki_T{t}_th{th}", A.kicked_ising(n, t, float(th), edges),
                        [A.parse_observable(s, n) for s in A.local_observables(n, edges)]))
    for t in cfg["bw_steps"]:
        out.append((f"bw_T{t}", A.brickwork(n, t, seed=t),
                    [A.parse_observable(s, n) for s in A.local_observables(n, chain)]))
    return out


def reached(values: Optional[Sequence[float]], exact: Sequence[float], cfg: Dict[str, Any]) -> bool:
    """Prereg section 6: every value finite, and a positive correlation whose square is at least
    the target; where the exact values have no spread, every value within flat_tol instead."""
    if values is None:
        return False
    v, x = np.asarray(values, float), np.asarray(exact, float)
    if v.shape != x.shape or not np.all(np.isfinite(v)):
        return False
    if np.std(x) <= 1e-12:
        return bool(np.max(np.abs(v - x)) <= cfg["flat_tol"])
    if np.std(v) <= 1e-12:
        return False
    r = float(np.corrcoef(v, x)[0, 1])
    return r > 0 and r * r >= cfg["target_r2"]


def judge_tier(naive: Dict[str, Dict[str, Any]], augur: Dict[str, Dict[str, Any]],
               exact: Dict[str, List[float]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """One tier in one repeat. Each arm maps circuit id to {"values": [...] or None, "cost_s": x}."""
    ids = list(exact)
    if set(naive) != set(ids) or set(augur) != set(ids):
        raise ValueError("both arms must report every circuit of the tier")
    rn = sum(reached(naive[c]["values"], exact[c], cfg) for c in ids)
    ra = sum(reached(augur[c]["values"], exact[c], cfg) for c in ids)
    mn = statistics.median(float(naive[c]["cost_s"]) for c in ids)
    ma = statistics.median(float(augur[c]["cost_s"]) for c in ids)
    return {"naive_reached": rn, "augur_reached": ra, "naive_median_s": round(mn, 4),
            "augur_median_s": round(ma, 4), "passes": ra >= rn and ma < mn}


def verdicts(repeats: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, str]:
    """SAVES for a tier only when it passes in every repeat the protocol asks for."""
    out = {}
    for tier in cfg["tiers"]:
        rows = [r["tiers"][tier]["judged"] for r in repeats if tier in r.get("tiers", {})]
        ok = len(rows) == cfg["repeats"] and all(j["passes"] for j in rows)
        out[tier] = "SAVES" if ok else "NO SAVINGS CLAIM"
    return out


# ── children ─────────────────────────────────────────────────────────────────────────────────────

def _limit_memory() -> None:   # pragma: no cover - runs in the child
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (CHILD_MEMORY_BYTES, CHILD_MEMORY_BYTES))


def _child(cfg: Dict[str, Any], args: Sequence[str],
           timeout_s: float) -> Tuple[Optional[Dict[str, Any]], str, float]:
    """Run this file in a child with the protocol's thread count. Returns (result, why, wall
    seconds); why starts with "killed" only when the child ran past timeout_s."""
    env = {**os.environ, "OMP_NUM_THREADS": str(cfg["threads"]), "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, str(Path(__file__).resolve()), *args, "--cfg", json.dumps(cfg)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False,
                           preexec_fn=_limit_memory if os.name == "posix" else None)
    except subprocess.TimeoutExpired:
        return None, f"killed at {timeout_s}s", time.perf_counter() - t0
    wall = time.perf_counter() - t0
    lines = [ln[len(SENTINEL):] for ln in p.stdout.splitlines() if ln.startswith(SENTINEL)]
    if p.returncode != 0 or len(lines) != 1:
        return None, f"failed, exit {p.returncode}: {p.stderr.strip()[-300:]}", wall
    return json.loads(lines[0]), "", wall


def _num(v: Optional[float]) -> float:
    """An engine value; None (no value) becomes NaN, which never reaches the target."""
    return float("nan") if v is None else float(v)


def _child_reference(cfg: Dict[str, Any], tier: str) -> Dict[str, Any]:
    return {cid: [_num(v) for v in A.ExactEngine().run(qc, ops).values]
            for cid, qc, ops in circuits(cfg, tier)}


def _child_naive(cfg: Dict[str, Any], tier: str, idx: int) -> Dict[str, Any]:
    cid, qc, ops = circuits(cfg, tier)[idx]
    r = A.MPSEngine().run(qc, ops, int(cfg["naive_chi"]))
    return {"circuit": cid, "elapsed_s": float(r.elapsed_s), "values": [_num(v) for v in r.values]}


def _child_augur(cfg: Dict[str, Any], tier: str) -> Dict[str, Any]:
    cs = circuits(cfg, tier)
    camp = A.Campaign({cid: (qc, ops) for cid, qc, ops in cs}, A.local_engines(),
                      budget=float(cfg["budget_per_circuit_s"]) * len(cs), cost_metric="seconds",
                      lightcone=False, log_path=None)
    t0 = time.perf_counter()
    rep = camp.run()
    wall = time.perf_counter() - t0
    rung_s = {cid: float(sum(h.get("amount") or 0.0 for h in camp.history if h["circuit"] == cid))
              for cid, _, _ in cs}
    # the campaign's own time (planning, fusion, bookkeeping) is a cost too: shared out equally
    overhead = max(0.0, wall - sum(rung_s.values()))
    out: Dict[str, Any] = {"spent": rep["spent"], "steps": rep["steps"], "wall_s": wall,
                           "overhead_s": overhead,
                           "failures": [str(f)[:200] for f in rep["failures"]], "circuits": {}}
    for cid, _, _ in cs:
        est = rep["circuits"][cid]["estimates"]
        out["circuits"][cid] = {
            "cost_s": rung_s[cid] + overhead / len(cs), "rung_s": rung_s[cid],
            "values": [_num(e.get("value")) for e in est],
            "labels": dict(A._count([str(e["basis"]) for e in est])),
            "ladders": rep["circuits"][cid]["ladders"]}
    return out


# ── the run ──────────────────────────────────────────────────────────────────────────────────────

def _load() -> float:
    return round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else -1.0


def _naive_arm(cfg: Dict[str, Any], tier: str, n_circuits: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    arm: Dict[str, Any] = {}
    meta: Dict[str, Any] = {"load_at_start": _load(), "cuts": []}
    for i in range(n_circuits):
        res, why, wall = _child(cfg, ["_naive", tier, str(i)], float(cfg["naive_cap_s"]))
        cid = circuits(cfg, tier)[i][0]
        if res is None and why.startswith("killed"):
            arm[cid] = {"values": None, "cost_s": float(cfg["naive_cap_s"]), "cut": why}
            meta["cuts"].append(cid)
        elif res is None:      # crashed before the cap: what it spent, and not reached
            arm[cid] = {"values": None, "cost_s": round(wall, 4), "failed": why}
            meta.setdefault("failed", []).append(cid)
        else:
            arm[cid] = {"values": res["values"], "cost_s": res["elapsed_s"]}
    return arm, meta


def _augur_arm(cfg: Dict[str, Any], tier: str, ids: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {"load_at_start": _load()}
    res, why, _ = _child(cfg, ["_augur", tier], float(cfg["augur_kill_s"]))
    unreached = {"values": None, "cost_s": float(cfg["budget_per_circuit_s"])}
    if res is None:            # killed or crashed: every circuit unreached at its budget
        meta["killed" if why.startswith("killed") else "failed"] = why
        return {c: dict(unreached) for c in ids}, meta
    meta.update({k: res[k] for k in ("spent", "steps", "failures", "wall_s", "overhead_s")})
    arm = {c: res["circuits"].get(c) or {**unreached, "missing": True} for c in ids}
    return arm, meta


def run(cfg: Dict[str, Any], log=print) -> Dict[str, Any]:
    reference: Dict[str, Dict[str, List[float]]] = {}
    for tier in cfg["tiers"]:
        ref, why, _ = _child(cfg, ["_reference", tier], REFERENCE_TIMEOUT_S)
        if ref is None:
            raise RuntimeError(f"exact reference for tier {tier} failed: {why}")
        reference[tier] = ref
        log(f"reference {tier}: {len(ref)} circuits")
    repeats = []
    for r in range(1, int(cfg["repeats"]) + 1):
        rec: Dict[str, Any] = {"repeat": r, "tiers": {}}
        for tier in cfg["tiers"]:
            ids = list(reference[tier])
            order = ("augur", "naive") if r % 2 == 0 else ("naive", "augur")
            arms: Dict[str, Any] = {}
            for arm in order:
                arms[arm] = (_naive_arm(cfg, tier, len(ids)) if arm == "naive"
                             else _augur_arm(cfg, tier, ids))
            judged = judge_tier(arms["naive"][0], arms["augur"][0], reference[tier], cfg)
            rec["tiers"][tier] = {"order": list(order), "naive": arms["naive"][0],
                                  "naive_meta": arms["naive"][1], "augur": arms["augur"][0],
                                  "augur_meta": arms["augur"][1], "judged": judged}
            log(f"repeat {r} tier {tier}: {judged}")
        repeats.append(rec)
    return {"reference": reference,
            "reference_sha256": {t: reference_digest(v) for t, v in reference.items()},
            "repeats": repeats, "verdicts": verdicts(repeats, cfg)}


def reference_digest(ref: Dict[str, List[float]]) -> str:
    """SHA-256 of a tier's exact values, canonical JSON (sorted keys, repr floats)."""
    return hashlib.sha256(json.dumps(ref, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ── the seal ─────────────────────────────────────────────────────────────────────────────────────

def _sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def ots_digest(path: Path) -> Optional[str]:
    """The SHA-256 a detached OpenTimestamps proof commits to (magic, version 1, op sha256, digest).
    That the proof is anchored in Bitcoin is checked with `ots verify`, not here."""
    if not path.is_file():
        return None
    b = path.read_bytes()
    n = len(OTS_MAGIC)
    if b[:n] != OTS_MAGIC or len(b) < n + 34 or b[n] != 0x01 or b[n + 1] != 0x08:
        return None
    return b[n + 2:n + 34].hex()


def result_path() -> Path:
    return RESULTS / f"RESULT_{PREREG_SHA256[:12]}.json"


def seal_state() -> Dict[str, Any]:
    have = _sha(PREREG)
    refusals = []
    if have != PREREG_SHA256:
        refusals.append(f"prereg digest {have} is not the pinned {PREREG_SHA256}")
    od = ots_digest(OTS)
    if od is None:
        refusals.append(f"no readable OpenTimestamps proof at {OTS.name}")
    elif od != have:
        refusals.append(f"the OpenTimestamps proof commits to {od}, not the prereg's {have}")
    if result_path().exists():
        refusals.append(f"{result_path().name} exists; the sealed run is done once")
    return {"prereg_sha256": have, "pinned": PREREG_SHA256, "ots_digest": od, "refusals": refusals}


def _versions() -> Dict[str, str]:
    import qiskit
    import qiskit_aer
    out = {"python": platform.python_version(), "machine": platform.machine(),
           "numpy": np.__version__, "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
           "cpus": str(os.cpu_count())}
    try:
        out["augur_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["augur_commit"] = "unknown"
    return out


def run_sealed(log=print) -> Path:
    st = seal_state()
    if st["refusals"]:
        raise RuntimeError("sealed run refused: " + "; ".join(st["refusals"]))
    lock = result_path().with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:                       # one sealed run at a time: a second start fails here, not hours later
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"sealed run refused: {lock.name} exists (another run holds it, or a "
                           f"crashed run left it; remove it by hand after checking)") from None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = run(SEALED, log=log)
    body = {"kind": "augur_cost_test", "prereg_sha256": PREREG_SHA256,
            "harness_sha256": _sha(Path(__file__).resolve()), "protocol": SEALED,
            "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": _versions(), **out}
    path = result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    lock.unlink()
    return path


def verify(path: Path, *, recompute_reference: bool = False) -> Dict[str, Any]:
    """Re-decide a result from what it recorded: the protocol is the sealed one, every repeat and
    circuit is present, the reference matches its recorded digest (and, with recompute_reference,
    the exact values computed again here, minutes at 24 qubits), the target is recomputed from the
    recorded values and the rule from the recorded costs. A verdict that does not follow is a
    problem. The costs themselves cannot be re-timed; that is what the three repeats are for."""
    problems: List[str] = []
    notes: List[str] = []
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        _verify_body(body, problems, notes, recompute_reference)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        problems.append(f"unreadable or malformed result: {type(e).__name__}: {e}")
        body = {}
    return {"ok": not problems, "problems": problems, "notes": notes,
            "verdicts": body.get("verdicts") if isinstance(body, dict) else None}


def _verify_body(body: Dict[str, Any], problems: List[str], notes: List[str],
                 recompute_reference: bool) -> None:
    cfg = SEALED
    if body.get("kind") != "augur_cost_test":
        problems.append(f"not a cost test result: {body.get('kind')!r}")
    if body.get("prereg_sha256") != PREREG_SHA256:
        problems.append(f"result is for prereg {body.get('prereg_sha256')}, not {PREREG_SHA256}")
    if body.get("protocol") != cfg:
        problems.append("the result was run under another protocol")
    if body.get("harness_sha256") != _sha(Path(__file__).resolve()):
        notes.append("this harness differs from the one that ran; the recorded numbers decide")
    ids = circuits_ids(cfg)
    for tier in cfg["tiers"]:
        ref = body["reference"].get(tier)
        if ref is None or sorted(ref) != sorted(ids):
            problems.append(f"tier {tier} reference does not hold exactly the protocol's circuits")
            continue
        if body.get("reference_sha256", {}).get(tier) != reference_digest(ref):
            problems.append(f"tier {tier} reference does not match its recorded digest")
        if recompute_reference:
            again, why, _ = _child(cfg, ["_reference", tier], REFERENCE_TIMEOUT_S)
            if again is None:
                problems.append(f"tier {tier} reference could not be recomputed: {why}")
            elif any(not np.allclose(again[c], ref[c], atol=1e-9, rtol=0) for c in ids):
                problems.append(f"tier {tier} reference differs from the exact values computed now")
    repeats = body.get("repeats", [])
    if [r.get("repeat") for r in repeats] != list(range(1, cfg["repeats"] + 1)):
        problems.append(f"{len(repeats)} repeats recorded, the protocol takes {cfg['repeats']} in order")
    for rec in repeats:
        for tier in cfg["tiers"]:
            t = rec.get("tiers", {}).get(tier)
            if t is None:
                problems.append(f"repeat {rec.get('repeat')} has no tier {tier}")
                continue
            try:
                again_j = judge_tier(t["naive"], t["augur"], body["reference"][tier], cfg)
            except ValueError as e:
                problems.append(f"repeat {rec.get('repeat')} tier {tier}: {e}")
                continue
            if again_j != t["judged"]:
                problems.append(f"repeat {rec.get('repeat')} tier {tier}: recorded {t['judged']}, "
                                f"recomputed {again_j}")
    if not problems and verdicts(repeats, cfg) != body.get("verdicts"):
        problems.append(f"verdicts {body.get('verdicts')} do not follow from the repeats")


def circuits_ids(cfg: Dict[str, Any]) -> List[str]:
    return ([f"ki_T{t}_th{th}" for t in cfg["ki_steps"] for th in cfg["ki_thetas"]]
            + [f"bw_T{t}" for t in cfg["bw_steps"]])


#: A tiny protocol for `dev` and the tests: 8 qubits, one repeat, short caps. Never a result.
DEV: Dict[str, Any] = {**SEALED, "tiers": {"S": [2, 4]}, "ki_steps": [3], "ki_thetas": [0.7],
                       "bw_steps": [3], "naive_chi": 16, "naive_cap_s": 60.0,
                       "budget_per_circuit_s": 5.0, "augur_kill_s": 120.0, "repeats": 1}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cost_test", description="Augur cost test")
    ap.add_argument("cmd", choices=["seal", "dev", "run", "verify", "_reference", "_naive", "_augur"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--reference", action="store_true", help="verify: recompute the exact answers")
    a = ap.parse_args(argv)
    if a.cmd.startswith("_"):
        cfg = json.loads(a.cfg) if a.cfg else SEALED
        out = (_child_reference(cfg, a.rest[0]) if a.cmd == "_reference"
               else _child_naive(cfg, a.rest[0], int(a.rest[1])) if a.cmd == "_naive"
               else _child_augur(cfg, a.rest[0]))
        print(SENTINEL + json.dumps(out))
        return 0
    if a.cmd == "seal":
        st = seal_state()
        print(json.dumps(st, indent=1))
        return 1 if st["refusals"] else 0
    if a.cmd == "dev":
        out = run(DEV)
        print("DEV, not a result:", json.dumps(out["verdicts"]))
        return 0
    if a.cmd == "run":
        try:
            print("sealed result:", run_sealed())
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 2
        return 0
    res = verify(Path(a.rest[0]), recompute_reference=a.reference)
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
