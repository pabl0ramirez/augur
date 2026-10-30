"""The cost test harness (benchmarks/cost_test.py): the target and the rule decide what the
prereg says, a cut is a real kill, the seal refuses, and verify catches an edited result."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cost_test", Path(__file__).resolve().parent.parent / "benchmarks" / "cost_test.py")
CT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CT)

CFG = CT.SEALED
X = [0.9, -0.2, 0.4, 0.1, -0.7]


# ── the target ──────────────────────────────────────────────────────────────────────────────────

def test_the_target_is_a_positive_correlation_squared_at_least_the_bar():
    assert CT.reached(X, X, CFG)
    assert CT.reached([v * 0.5 + 0.01 for v in X], X, CFG)          # correlation is scale-free
    assert not CT.reached([-v for v in X], X, CFG)                   # r = -1 squares to 1: refused


def test_the_target_refuses_missing_nan_short_or_flat_estimates():
    assert not CT.reached(None, X, CFG)
    assert not CT.reached([*X[:-1], math.nan], X, CFG)
    assert not CT.reached(X[:-1], X, CFG)
    assert not CT.reached([0.0] * len(X), X, CFG)                    # a truncated zero has no spread


def test_the_target_is_tight_at_the_bar():
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    for noise, want in ((0.05, True), (0.2, False)):                 # r^2 about 0.9975 and 0.96
        assert CT.reached(list(x + noise * rng.standard_normal(400)), list(x), CFG) is want


def test_flat_exact_values_are_judged_by_distance():
    flat = [0.5] * 4
    assert CT.reached([0.505, 0.495, 0.5, 0.508], flat, CFG)
    assert not CT.reached([0.52, 0.5, 0.5, 0.5], flat, CFG)


# ── the rule ────────────────────────────────────────────────────────────────────────────────────

def _arm(costs, ok):
    return {f"c{i}": {"values": (X if good else None), "cost_s": c}
            for i, (c, good) in enumerate(zip(costs, ok, strict=True))}


EXACT = {f"c{i}": X for i in range(3)}


@pytest.mark.parametrize("augur_costs,augur_ok,passes", [
    ([1, 1, 1], [1, 1, 1], True),
    ([1, 1, 1], [1, 1, 0], False),      # cheaper but reaches fewer: refused
    ([5, 5, 5], [1, 1, 1], False),      # same median: not lower, refused
    ([9, 9, 9], [1, 1, 1], False),
])
def test_a_tier_passes_only_when_augur_is_as_accurate_and_strictly_cheaper(augur_costs, augur_ok, passes):
    j = CT.judge_tier(_arm([5, 5, 5], [1, 1, 1]), _arm(augur_costs, augur_ok), EXACT, CFG)
    assert j["passes"] is passes


def test_a_tier_refuses_an_arm_missing_a_circuit():
    with pytest.raises(ValueError, match="every circuit"):
        CT.judge_tier(_arm([5, 5], [1, 1]), _arm([1, 1, 1], [1, 1, 1]), EXACT, CFG)


def test_saves_needs_every_repeat_and_every_repeat_to_pass():
    ok = {"tiers": {"S": {"judged": {"passes": True}}, "L": {"judged": {"passes": True}}}}
    bad = {"tiers": {"S": {"judged": {"passes": True}}, "L": {"judged": {"passes": False}}}}
    assert CT.verdicts([ok, ok, ok], CFG) == {"S": "SAVES", "L": "SAVES"}
    assert CT.verdicts([ok, bad, ok], CFG) == {"S": "SAVES", "L": "NO SAVINGS CLAIM"}
    assert CT.verdicts([ok, ok], CFG) == {"S": "NO SAVINGS CLAIM", "L": "NO SAVINGS CLAIM"}


def test_the_circuits_are_the_bench_grid_at_both_sizes():
    for tier, n in (("S", 18), ("L", 24)):
        cs = CT.circuits(CFG, tier)
        assert [c for c, _, _ in cs] == CT.circuits_ids(CFG) and len(cs) == 12
        assert all(qc.num_qubits == n for _, qc, _ in cs)


# ── children ────────────────────────────────────────────────────────────────────────────────────

def test_a_naive_run_past_its_cap_is_killed_and_counted_at_the_cap():
    """NULL: the cap is a real kill, the cost recorded is the cap, and the circuit is not reached."""
    cfg = {**CT.DEV, "naive_cap_s": 0.5}
    arm, meta = CT._naive_arm(cfg, "S", 1)
    (row,) = arm.values()
    assert row["values"] is None and row["cost_s"] == 0.5 and "killed" in row["cut"]
    assert meta["cuts"] == list(arm)


def test_a_killed_campaign_counts_every_circuit_unreached_at_its_budget():
    cfg = {**CT.DEV, "augur_kill_s": 0.5}
    ids = CT.circuits_ids(cfg)
    arm, meta = CT._augur_arm(cfg, "S", ids)
    assert "killed" in meta["killed"]
    assert all(arm[c] == {"values": None, "cost_s": cfg["budget_per_circuit_s"]} for c in ids)


# ── the seal and verify ─────────────────────────────────────────────────────────────────────────

def test_the_prereg_is_the_pinned_one():
    assert CT._sha(CT.PREREG) == CT.PREREG_SHA256


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("protocol\n")
    sha = CT._sha(prereg)
    ots = tmp_path / "PREREG.md.ots"
    ots.write_bytes(CT.OTS_MAGIC + bytes([1, 8]) + bytes.fromhex(sha) + b"\xf0rest")
    monkeypatch.setattr(CT, "PREREG", prereg)
    monkeypatch.setattr(CT, "OTS", ots)
    monkeypatch.setattr(CT, "PREREG_SHA256", sha)
    monkeypatch.setattr(CT, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(CT, "SEALED", CT.DEV)
    return tmp_path


@pytest.mark.parametrize("damage", ["prereg", "no_proof", "foreign_proof", "result_exists"])
def test_the_sealed_run_refuses(sealed, damage):
    if damage == "prereg":
        CT.PREREG.write_text("protocol, moved\n")
    elif damage == "no_proof":
        CT.OTS.unlink()
    elif damage == "foreign_proof":
        CT.OTS.write_bytes(CT.OTS_MAGIC + bytes([1, 8]) + b"\xab" * 32)
    else:
        CT.result_path().parent.mkdir(parents=True)
        CT.result_path().write_text("{}")
    with pytest.raises(RuntimeError, match="refused"):
        CT.run_sealed(log=lambda *_: None)


@pytest.fixture
def result(sealed):
    return CT.run_sealed(log=lambda *_: None)


def test_a_sealed_run_verifies_and_runs_only_once(result):
    res = CT.verify(result)
    assert res["ok"], res
    with pytest.raises(RuntimeError, match="done once"):
        CT.run_sealed(log=lambda *_: None)


def _edit(path, fn):
    body = json.loads(path.read_text())
    fn(body)
    path.write_text(json.dumps(body))
    return CT.verify(path)


def test_verify_catches_an_edited_cost(result):
    def cheaper(b):
        for c in b["repeats"][0]["tiers"]["S"]["augur"].values():
            c["cost_s"] = 0.0
    assert any("recomputed" in p for p in _edit(result, cheaper)["problems"])


def test_verify_catches_an_edited_value(result):
    def wrong(b):
        for c in b["repeats"][0]["tiers"]["S"]["naive"].values():
            c["values"] = [0.0] * len(c["values"])
    assert any("recomputed" in p for p in _edit(result, wrong)["problems"])


def test_verify_catches_a_verdict_that_does_not_follow(result):
    def flip(b):
        b["verdicts"]["S"] = "SAVES" if b["verdicts"]["S"] != "SAVES" else "NO SAVINGS CLAIM"
    assert any("do not follow" in p for p in _edit(result, flip)["problems"])


def test_verify_catches_a_dropped_repeat_or_another_protocol(result):
    assert any("repeats recorded" in p for p in _edit(result, lambda b: b["repeats"].clear())["problems"])
    assert any("another protocol" in p
               for p in _edit(result, lambda b: b["protocol"].update(naive_cap_s=1e9))["problems"])


def test_a_naive_run_that_crashes_before_the_cap_costs_its_time_and_is_not_reached():
    """NULL: a crash is not a cut. It is charged what it spent, never the cap, and never reached."""
    cfg = {**CT.DEV, "naive_chi": 0}                       # the engine refuses bond dimension 0
    arm, meta = CT._naive_arm(cfg, "S", 1)
    (row,) = arm.values()
    assert row["values"] is None and "failed" in row and row["cost_s"] < cfg["naive_cap_s"]
    assert meta["cuts"] == [] and meta["failed"] == list(arm)


def test_a_crashed_campaign_counts_every_circuit_unreached_at_its_budget():
    cfg = {**CT.DEV, "budget_per_circuit_s": -1.0}         # Campaign refuses a negative budget
    ids = CT.circuits_ids(cfg)
    arm, meta = CT._augur_arm(cfg, "S", ids)
    assert "failed" in meta and all(arm[c]["values"] is None for c in ids)
    assert all(arm[c]["cost_s"] == -1.0 for c in ids)


def test_augur_is_charged_its_own_overhead_not_only_its_rungs(result):
    body = json.loads(result.read_text())
    t = body["repeats"][0]["tiers"]["S"]
    assert t["augur_meta"]["overhead_s"] >= 0
    for c in t["augur"].values():
        assert c["cost_s"] >= c["rung_s"]
    assert sum(c["cost_s"] for c in t["augur"].values()) == pytest.approx(t["augur_meta"]["wall_s"])


def test_a_second_sealed_run_is_refused_while_one_holds_the_lock(sealed):
    lock = CT.result_path().with_suffix(".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("12345")
    with pytest.raises(RuntimeError, match="lock"):
        CT.run_sealed(log=lambda *_: None)


def test_verify_catches_an_edited_reference(result):
    def shift(b):
        for vals in b["reference"]["S"].values():
            vals[0] += 0.5
    assert any("recorded digest" in p for p in _edit(result, shift)["problems"])


def test_verify_recomputes_the_reference_when_asked(result):
    """NULL: rewrite the reference AND its digest; only recomputing the exact values catches it."""
    def forge(b):
        for vals in b["reference"]["S"].values():
            vals[0] += 0.5
        b["reference_sha256"]["S"] = CT.reference_digest(b["reference"]["S"])
    _edit(result, forge)
    assert not any("reference" in p for p in CT.verify(result)["problems"])
    assert any("computed now" in p for p in CT.verify(result, recompute_reference=True)["problems"])


def test_verify_reports_a_malformed_result_instead_of_crashing(tmp_path, sealed):
    bad = tmp_path / "bad.json"
    bad.write_text('{"kind": "augur_cost_test", "repeats": [{"repeat": 1}]}')
    res = CT.verify(bad)
    assert not res["ok"] and res["problems"]
