"""Tests for src/augur.py. Every engine is checked against exact statevector on circuits
small enough to know the answer, including mixed-letter observables that expose bit-order
mistakes; every guard is shown to fire."""
import math
from pathlib import Path

import numpy as np
import pytest
from typing import ClassVar
from qiskit import QuantumCircuit

from augur import core as A


def _ki(n=8, steps=4, th=0.7):
    edges = [(i, i + 1) for i in range(n - 1)]
    return A.kicked_ising(n, steps, th, edges), edges


def _clifford_mix(n=6, seed=3):
    """Cliffords (h, s, cx, cz) mixed with rotations, so the Clifford-prefix path of
    pauli-prop and Y observables are exercised."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for layer in range(4):
        for q in range(n):
            qc.h(q) if (q + layer) % 2 else qc.s(q)
            qc.ry(float(rng.uniform(0, 3)), q)
        for q in range(layer % 2, n - 1, 2):
            qc.cx(q, q + 1) if layer % 2 else qc.cz(q, q + 1)
        qc.rz(float(rng.uniform(0, 3)), layer % n)
    return qc


MIXED = ["Z0X1", "X0Y1Z2", "Y2X3", "0.5*X0Z1-Y4", "Z5"]


# ── observables and bit order ────────────────────────────────────────────────
def test_parse_observable_qiskit_order():
    assert A.parse_observable("Z0", 3).to_list()[0][0] == "IIZ"   # rightmost = qubit 0
    assert A.parse_observable("X0Y2", 3).to_list()[0][0] == "YIX"


def test_parse_observable_sum_coefficients_and_scientific_notation():
    d = dict((lbl, c.real) for lbl, c in A.parse_observable("0.5*Z0+Z1-X2", 3).to_list())
    assert d == {"IIZ": 0.5, "IZI": 1.0, "XII": -1.0}
    d = dict((lbl, c.real) for lbl, c in A.parse_observable("1e-3*Z0-2.5E-1*X1", 2).to_list())
    assert d == {"IZ": pytest.approx(1e-3), "XI": pytest.approx(-0.25)}


@pytest.mark.parametrize("bad", ["Q0", "Z", "Z9", "", "Z0Z0", "+"])
def test_parse_observable_refuses_bad_specs(bad):
    with pytest.raises(ValueError):
        A.parse_observable(bad, 3)


def test_bq_pauli_sum_matches_measured_bluequbit_order():
    # job VTmH4MQbNjbBf6jC: X on qubit 0 gave <IIZ> = -1, so "Z0" must map to "IIZ"
    assert A.to_bq_pauli_sum(A.parse_observable("Z0", 3)) == [("IIZ", 1.0)]


# ── engines agree with exact, mixed letters included ─────────────────────────
def test_lightcone_mixed_letter_reviewer_repro():
    """The reviewer's repro: Z0 X1 after ry(0.5,1); ry(0.4,0); rx(0.7,0).
    Exact is cos0.4 cos0.7 sin0.5; the unreversed label gave cos0.4 sin0.5."""
    qc = QuantumCircuit(2)
    qc.ry(0.5, 1)
    qc.ry(0.4, 0)
    qc.rx(0.7, 0)
    op = [A.parse_observable("Z0X1", 2)]
    want = math.cos(0.4) * math.cos(0.7) * math.sin(0.5)
    assert A.LightConeEngine().run(qc, op).values[0] == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("make", [lambda: _ki()[0], lambda: _clifford_mix(8), lambda: A.brickwork(8, 4, 2)])
def test_every_engine_matches_exact_on_mixed_observables(make):
    qc = make()
    ops = [A.parse_observable(s, qc.num_qubits) for s in MIXED]
    ex = np.array(A.ExactEngine().run(qc, ops).values)
    assert np.abs(np.array(A.MPSEngine().run(qc, ops, 64).values) - ex).max() < 1e-8
    assert np.abs(np.array(A.PPSEngine().run(qc, ops, 1e-7).values) - ex).max() < 1e-4
    assert np.abs(np.array(A.LightConeEngine().run(qc, ops).values, float) - ex).max() < 1e-10


# prepatch-ok: every local engine already refused mid-circuit measurement via _prepare; this
# round extends the assertion from MPS to all four engines (coverage, not new behaviour)
def test_final_measurements_are_stripped_and_mid_circuit_refused():
    qc = _clifford_mix(4)
    ops = [A.parse_observable("X0Y1", 4)]
    ref = A.ExactEngine().run(qc, ops).values
    m = qc.copy()
    m.measure_all()
    for eng, p in ((A.ExactEngine(), None), (A.MPSEngine(), 16), (A.PPSEngine(), 1e-7),
                   (A.LightConeEngine(), None)):
        assert eng.run(m, ops, p).values[0] == pytest.approx(ref[0], abs=1e-6)
    mid = QuantumCircuit(2, 1)
    mid.h(0)
    mid.measure(0, 0)
    mid.h(0)                       # a gate after the measurement makes it mid-circuit
    for eng, p in ((A.ExactEngine(), None), (A.MPSEngine(), 8), (A.PPSEngine(), 1e-3),
                   (A.LightConeEngine(), None)):
        with pytest.raises(ValueError, match="mid-circuit"):
            eng.run(mid, [A.parse_observable("Z0", 2)], p)


def test_lightcone_returns_none_when_cone_too_wide():
    qc, _ = _ki(n=6, steps=3)
    r = A.LightConeEngine(max_qubits=2).run(qc, [A.parse_observable("Z3", 6)])
    assert r.values == [None] and r.degenerate == [True]


def test_exact_engine_refuses_large_circuit(monkeypatch):
    monkeypatch.setattr(A, "EXACT_MAX_QUBITS", 4)
    qc, _ = _ki(n=6)
    with pytest.raises(ValueError):
        A.ExactEngine().run(qc, [A.parse_observable("Z0", 6)])


def test_truncated_mps_is_measurably_wrong():
    qc, _ = _ki(n=10, steps=6, th=0.9)
    ops = [A.parse_observable(f"Z{q}", 10) for q in range(10)]
    ex = np.array(A.ExactEngine().run(qc, ops).values)
    assert np.abs(np.array(A.MPSEngine().run(qc, ops, 2).values) - ex).max() > 1e-3


def test_pps_marks_empty_operator_and_term_cap_degenerate():
    qc, _ = _ki(n=8, steps=6, th=0.9)
    ops = [A.parse_observable("Z3", 8)]
    coarse = A.PPSEngine().run(qc, ops, 0.9)            # threshold above every coefficient
    assert coarse.values == [0.0] and coarse.degenerate == [True]
    capped = A.PPSEngine(max_terms=4).run(qc, ops, 1e-6)
    assert capped.degenerate == [True]
    fine = A.PPSEngine().run(qc, ops, 1e-6)
    assert fine.degenerate == [False]


def test_pps_terms_cap_bounded_by_memory():
    assert A.PPSEngine().terms_cap(100) <= A.PPS_MEMORY_BYTES // 216
    assert A.PPSEngine(max_terms=7).terms_cap(100) == 7


# ── BlueQubit engine against a client that applies the real SDK's own check ──
class _Res:
    def __init__(self, ev, cost=0.01, job_id="J"):
        self.expectation_value, self.cost, self.job_id = ev, cost, job_id


class _Client:
    """Fake client. For pauli-path it calls the SDK's own client-side validator
    (BQClient._validate_pps_job, a private method of SDK 0.18.12b1), so a call the SDK would
    refuse before sending fails here too; server-side checks are not reproduced. Every job is
    stored at run() and returned by wait()."""

    def __init__(self, evs):
        self.evs, self.calls, self.k, self.jobs, self.cancelled = list(evs), [], 0, {}, []

    def run(self, target, device, pauli_sum=None, options=None, job_name=None, asynchronous=False):
        from bluequbit import BQClient

        if device == "pauli-path":
            BQClient._validate_pps_job(options, target, pauli_sum)
        self.calls.append({"device": device, "pauli_sum": pauli_sum, "options": options,
                           "asynchronous": asynchronous})
        self.k += 1
        ev = self.evs[self.k - 1] if device.startswith("pauli-path") else self.evs
        res = _Res(ev, job_id=f"J{self.k}")
        self.jobs[res.job_id] = res
        return res

    def wait(self, job_ids, timeout=None):
        return [self.jobs[j] for j in job_ids]

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def test_real_sdk_refuses_two_observables_on_pauli_path():
    from bluequbit import BQClient

    with pytest.raises(Exception, match="single observable"):
        BQClient._validate_pps_job({"pauli_path_truncation_threshold": 1e-3}, QuantumCircuit(2),
                                   [[("IZ", 1.0)], [("ZI", 1.0)]])


def test_bluequbit_pauli_path_one_job_per_observable():
    cl = _Client([0.1, -0.2])
    eng = A.BlueQubitEngine(cl, "pauli-path")
    r = eng.run(QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 1e-3)
    assert r.values == [0.1, -0.2] and r.cost_usd == pytest.approx(0.02)
    assert [c["pauli_sum"] for c in cl.calls] == [[("IZ", 1.0)], [("ZI", 1.0)]]
    assert all(c["asynchronous"] for c in cl.calls)


def test_bluequbit_mps_one_job_all_observables():
    cl = _Client([0.3, 0.4])
    r = A.BlueQubitEngine(cl, "mps.cpu").run(QuantumCircuit(2),
                                             [A.parse_observable("Z0", 2), A.parse_observable("X1", 2)], 64)
    assert r.values == [0.3, 0.4] and len(cl.calls) == 1
    assert cl.calls[0]["options"] == {"mps_bond_dimension": 64}


def test_bluequbit_engine_guards_fire():
    with pytest.raises(ValueError):
        A.BlueQubitEngine(_Client([0]), "pauli-path").options_for(1e-6)
    with pytest.raises(ValueError):
        A.BlueQubitEngine(_Client([0]), "mps.cpu").options_for(4096)
    assert A.BlueQubitEngine(_Client([0]), "mps.gpu").options_for(4000) == {"mps_bond_dimension": 4000}
    with pytest.raises(RuntimeError):
        A.BlueQubitEngine(_Client([0.1]), "mps.cpu").run(
            QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 64)


class _StatusClient(_Client):
    """wait() raises the SDK's own BQJobNotCompleteError (structured run_status) for the first
    `statuses`, one per call, the way BQClient.wait reports a job that ended in a no-result
    state (Duke 2026-10-24 live)."""

    def __init__(self, evs, statuses):
        super().__init__(evs)
        self.statuses = list(statuses)

    def wait(self, job_ids, timeout=None):
        from bluequbit.exceptions import BQJobNotCompleteError

        if self.statuses:
            raise BQJobNotCompleteError(job_ids[0], self.statuses.pop(0), "server message")
        return super().wait(job_ids, timeout)


def test_bluequbit_status_is_read_from_the_field_not_the_text():
    """Null test: an error whose TEXT names the cap state but carries no run_status is not a
    bounce (the review's point: substring matching misclassifies wrapped errors)."""
    class _TextOnly(_Client):
        def wait(self, job_ids, timeout=None):
            raise RuntimeError("upstream said HACKATHON_PARALLEL_JOBS_LIMIT somewhere in a log line")

    slept = []
    cl = _TextOnly([0.1])
    with pytest.raises(RuntimeError, match="failed or timed out"):
        A.BlueQubitEngine(cl, "pauli-path", sleep=slept.append).run(
            QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-3)
    assert slept == [] and len(cl.calls) == 1


def test_bluequbit_bounces_are_capped_without_a_deadline():
    """No timeout means an infinite deadline; a cap that never lifts must still end."""
    slept = []
    cl = _StatusClient([0.1] * 5, ["HACKATHON_PARALLEL_JOBS_LIMIT"] * 5)
    eng = A.BlueQubitEngine(cl, "pauli-path", max_bounces=2, sleep=slept.append)
    with pytest.raises(RuntimeError, match="PARALLEL_JOBS_LIMIT"):
        eng.run(QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-3)
    assert len(slept) == 2 and len(cl.calls) == 3


def test_bluequbit_not_enough_funds_is_a_refusal():
    cl = _StatusClient([0.1], ["NOT_ENOUGH_FUNDS"])
    with pytest.raises(ValueError, match="refused"):
        A.BlueQubitEngine(cl, "mps.gpu", sleep=lambda s: None).run(
            QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1024)


def test_bluequbit_extra_options_cannot_override_checked_keys():
    with pytest.raises(ValueError, match="not allowed"):
        A.BlueQubitEngine(_Client([0]), "mps.cpu", extra_options={"mps_bond_dimension": 4096})
    with pytest.raises(ValueError, match="not allowed"):
        A.BlueQubitEngine(_Client([0]), "pauli-path", extra_options={"pauli_path_truncation_threshold": 1e-9})
    with pytest.raises(ValueError, match="not allowed"):   # the weight cutoff is a GPU option
        A.BlueQubitEngine(_Client([0]), "pauli-path", extra_options={"pauli_path_weight_cutoff": 12})
    with pytest.raises(ValueError, match="bounce_wait_s"):
        A.BlueQubitEngine(_Client([0]), "pauli-path", bounce_wait_s=0)


def test_bluequbit_ctrl_c_during_submission_cancels_what_went_in():
    class _Interrupt(_Client):
        def run(self, *a, **k):
            if self.k == 1:
                raise KeyboardInterrupt
            return super().run(*a, **k)

    cl = _Interrupt([0.1, 0.2])
    with pytest.raises(KeyboardInterrupt):
        A.BlueQubitEngine(cl, "pauli-path").run(
            QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 1e-3)
    assert cl.cancelled == ["J1"]


def test_bluequbit_bounce_at_submission_resubmits():
    """The cap can refuse at submit time too (BQClient.run raises for a no-result state)."""
    from bluequbit.exceptions import BQJobNotCompleteError

    class _SubmitBounce(_Client):
        def run(self, *a, **k):
            if not getattr(self, "bounced", False):
                self.bounced = True
                raise BQJobNotCompleteError("J0", "HACKATHON_PARALLEL_JOBS_LIMIT", "cap")
            return super().run(*a, **k)

    slept = []
    r = A.BlueQubitEngine(_SubmitBounce([0.4]), "pauli-path", sleep=slept.append).run(
        QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-3)
    assert r.values == [0.4] and len(slept) == 1


def test_bluequbit_status_ignores_suppressed_context():
    from bluequbit.exceptions import BQJobNotCompleteError

    try:
        try:
            raise BQJobNotCompleteError("J", "HACKATHON_PARALLEL_JOBS_LIMIT", "old")
        except BQJobNotCompleteError:
            raise RuntimeError("unrelated") from None
    except RuntimeError as e:
        assert A.BlueQubitEngine._status(e) is None
    loop = RuntimeError("a")
    loop.__context__ = loop                                  # a cycle must end, not spin
    assert A.BlueQubitEngine._status(loop) is None


def test_bluequbit_engine_rejects_bad_values():
    for kw in ({"max_bounces": -1}, {"max_bounces": True}, {"timeout_s": 0}, {"timeout_s": float("nan")},
               {"bounce_wait_s": float("inf")}, {"extra_options": ["level"]},
               {"extra_options": {"pauli_path_circuit_transpilation_level": "high"}},
               {"extra_options": {"pauli_path_circuit_transpilation_level": 7}}):
        with pytest.raises(ValueError):
            A.BlueQubitEngine(_Client([0]), "pauli-path", **kw)
    with pytest.raises(ValueError, match="weight_cutoff"):
        A.BlueQubitEngine(_Client([0]), "pauli-path.gpu", extra_options={"pauli_path_weight_cutoff": -1})


def test_run_day_rejects_bad_ladders_and_stray_device_options(tmp_path):
    base = {"budget_usd": 1.0, "devices": ["mps.cpu"],
            "circuits": {"c": {"target": "id-c", "n_qubits": 2, "observables": ["Z0"]}}}
    for bad in ([64.5], ["1e3"], [True], [0], [float("inf")]):
        with pytest.raises(ValueError, match="whole numbers"):
            A.run_day({**base, "params": {"mps.cpu": bad}}, _Client([0.0]), out_dir=tmp_path)
    with pytest.raises(ValueError, match="positive finite"):
        A.run_day({**base, "devices": ["pauli-path"], "params": {"pauli-path": [float("nan")]}},
                  _Client([0.0]), out_dir=tmp_path)
    with pytest.raises(ValueError, match="does not run"):
        A.run_day({**base, "device_options": {"pauli-path": {"pauli_path_circuit_transpilation_level": 3}}},
                  _Client([0.0]), out_dir=tmp_path)
    with pytest.raises(ValueError, match="does not run"):            # params for an unused device
        A.run_day({**base, "params": {"pauli-path": [1e-3]}}, _Client([0.0]), out_dir=tmp_path)
    with pytest.raises(ValueError, match="ascending"):
        A.run_day({**base, "params": {"mps.cpu": [256, 64]}}, _Client([0.0]), out_dir=tmp_path)
    with pytest.raises(ValueError, match="descending"):
        A.run_day({**base, "devices": ["pauli-path"], "params": {"pauli-path": [1e-4, 1e-3]}},
                  _Client([0.0]), out_dir=tmp_path)
    for tol in (True, float("inf"), 0, "1e-3"):
        with pytest.raises(ValueError, match="eps_tol"):
            A.run_day({**base, "eps_tol": tol}, _Client([0.0]), out_dir=tmp_path)
    with pytest.raises(ValueError, match="mapping"):
        A.run_day({**base, "params": [64]}, _Client([0.0]), out_dir=tmp_path)


def test_bluequbit_cap_bounce_resubmits_then_succeeds():
    slept = []
    cl = _StatusClient([0.25, 0.25], ["HACKATHON_PARALLEL_JOBS_LIMIT"])
    eng = A.BlueQubitEngine(cl, "mps.cpu", bounce_wait_s=7.0, sleep=slept.append)
    r = eng.run(QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 64)
    assert r.values == [0.25, 0.25]
    assert len(cl.calls) == 2 and slept == [7.0]     # bounced once, resubmitted once
    assert cl.cancelled == ["J1"]                     # the bounced job was cancelled, not left


def test_bluequbit_gpu_cooldown_also_resubmits():
    slept = []
    cl = _StatusClient([0.1, 0.1], ["HACKATHON_GPU_QUOTA_COOLDOWN"])   # one value per submit
    r = A.BlueQubitEngine(cl, "pauli-path.gpu", sleep=slept.append).run(
        QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-3)
    assert r.values == [0.1] and len(slept) == 1 and len(cl.calls) == 2


def test_bluequbit_failed_validation_is_a_refusal_not_a_retry():
    """A threshold the server refuses can never succeed: ValueError retires the ladder at once
    (Campaign does not retry ValueError), and nothing is resubmitted."""
    slept = []
    cl = _StatusClient([0.1], ["FAILED_VALIDATION"])
    with pytest.raises(ValueError, match="refused"):
        A.BlueQubitEngine(cl, "pauli-path", sleep=slept.append).run(
            QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-4)
    assert len(cl.calls) == 1 and slept == []


def test_bluequbit_bounce_stops_at_deadline():
    """With no time left before the deadline a bounce is not resubmitted: it fails like any
    other cloud error, so the campaign's deadline is never overrun by waiting on the cap."""
    cl = _StatusClient([0.1], ["HACKATHON_PARALLEL_JOBS_LIMIT"])
    eng = A.BlueQubitEngine(cl, "pauli-path", timeout_s=1.0, bounce_wait_s=60.0, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="PARALLEL_JOBS_LIMIT"):
        eng.run(QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-3)
    assert len(cl.calls) == 1


def test_bluequbit_extra_options_reach_the_job():
    cl = _Client([0.2])
    A.BlueQubitEngine(cl, "pauli-path", extra_options={"pauli_path_circuit_transpilation_level": 3}).run(
        QuantumCircuit(2), [A.parse_observable("Z0", 2)], 1e-3)
    assert cl.calls[0]["options"] == {"pauli_path_truncation_threshold": 1e-3,
                                      "pauli_path_circuit_transpilation_level": 3}
    opts = A.BlueQubitEngine(_Client([0]), "pauli-path.gpu",
                             extra_options={"pauli_path_weight_cutoff": 12}).options_for(1e-4)
    assert opts["pauli_path_weight_cutoff"] == 12


def test_run_day_spec_params_and_device_options(tmp_path):
    cl = _Client([0.0] * 50)
    spec = {"budget_usd": 1.0, "devices": ["pauli-path"], "params": {"pauli-path": [1e-3, 3e-4]},
            "device_options": {"pauli-path": {"pauli_path_circuit_transpilation_level": 3}},
            "circuits": {"c": {"target": "snr-x", "n_qubits": 2, "observables": ["Z0"]}}}
    A.run_day(spec, cl, out_dir=tmp_path)
    thr = [c["options"]["pauli_path_truncation_threshold"] for c in cl.calls]
    assert thr and set(thr) <= {1e-3, 3e-4}                 # the spec's ladder, nothing finer
    assert all(c["options"]["pauli_path_circuit_transpilation_level"] == 3 for c in cl.calls)
    with pytest.raises(ValueError, match="does not run"):    # caught before the device lookup
        A.run_day({**spec, "params": {"quantum": [1]}}, cl, out_dir=tmp_path)


class _MPSLadderClient(_Client):
    """mps values by bond dimension: 64 -> 0.0010, 128 -> 0.0030, 256 -> 0.0031 (noise-level
    drift that a 1e-2 tolerance would wrongly call converged at 128)."""
    V: ClassVar[dict] = {64: 0.0010, 128: 0.0030, 256: 0.0031}

    def run(self, target, device, pauli_sum=None, options=None, job_name=None, asynchronous=False):
        self.evs = [self.V[options["mps_bond_dimension"]]]
        return super().run(target, device, pauli_sum, options, job_name, asynchronous)


def test_run_day_eps_tol_keeps_small_values_climbing(tmp_path):
    spec = {"budget_usd": 1.0, "devices": ["mps.cpu"], "params": {"mps.cpu": [64, 128, 256]},
            "circuits": {"c": {"target": "id-c", "n_qubits": 2, "observables": ["Z0"]}}}
    loose = _MPSLadderClient([0.0])
    A.run_day(spec, loose, out_dir=tmp_path / "loose")
    assert [c["options"]["mps_bond_dimension"] for c in loose.calls] == [64, 128]   # false stop
    tight = _MPSLadderClient([0.0])
    A.run_day({**spec, "eps_tol": 5e-4}, tight, out_dir=tmp_path / "tight")
    assert [c["options"]["mps_bond_dimension"] for c in tight.calls] == [64, 128, 256]
    with pytest.raises(ValueError, match="eps_tol"):
        A.run_day({**spec, "eps_tol": 0}, tight, out_dir=tmp_path / "zero")


# ── ladders ──────────────────────────────────────────────────────────────────
class _Scripted(A.Engine):
    method = "fake"

    def __init__(self, seq, secs, degen=None, fail_at=None):
        self.seq, self.secs, self.degen, self.fail_at, self.i = seq, secs, degen, fail_at, 0

    def run(self, target, ops, param):
        if self.fail_at is not None and self.i == self.fail_at:
            raise RuntimeError("cloud said no")
        v, s = self.seq[self.i], self.secs[self.i]
        d = self.degen[self.i] if self.degen else [False] * len(v)
        self.i += 1
        return A.Rung(self.method, param, list(v), s, degenerate=d)


def test_ladder_stops_on_convergence():
    lad = A.run_ladder(_Scripted([[0.5], [0.3], [0.301], [0.9]], [0.1] * 4), None, [None],
                       [1, 2, 3, 4], eps_tol=1e-2)
    assert len(lad.rungs) == 3 and lad.converged == [True] and "converged" in lad.stop_reason


def test_true_zero_converges_like_any_value():
    """An observable forbidden by symmetry is exactly 0 at every rung; it must converge, or the
    campaign pours budget into it forever (the bug the campaign test caught)."""
    lad = A.run_ladder(_Scripted([[0.0], [0.0], [0.42]], [0.1] * 3), None, [None], [1, 2, 3], eps_tol=1e-2)
    assert len(lad.rungs) == 2 and lad.converged == [True]


def test_cloud_pauli_path_zero_is_unverified():
    cl = _Client([0.0, 0.3])
    r = A.BlueQubitEngine(cl, "pauli-path").run(
        QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 1e-3)
    assert r.degenerate == [True, False]


def test_degenerate_rungs_never_converge():
    lad = A.run_ladder(_Scripted([[0.2], [0.2], [0.2]], [0.1] * 3, degen=[[True], [True], [False]]),
                       None, [None], [1, 2, 3], eps_tol=1e-2)
    assert lad.converged == [False]


def test_ladder_stops_on_predicted_overrun():
    # time doubles per rung; the fit predicts 4 s for the fourth, which overruns 3.6 s
    lad = A.run_ladder(_Scripted([[0.1], [0.5], [0.9], [0.2]], [0.5, 1.0, 2.0, 4.0]), None, [None],
                       [16, 32, 64, 128], eps_tol=1e-6, budget_s=3.6)
    assert len(lad.rungs) == 3 and "predicted" in lad.stop_reason and lad.converged == [False]


def test_ladder_stops_when_budget_spent():
    lad = A.run_ladder(_Scripted([[0.1], [0.5]], [5.0, 5.0]), None, [None], [16, 32], budget_s=4.0)
    assert len(lad.rungs) == 1 and "spent" in lad.stop_reason


def test_engine_failure_keeps_finished_rungs():
    lad = A.run_ladder(_Scripted([[0.1], [0.2], [0.3]], [0.1] * 3, fail_at=2), None, [None],
                       [1, 2, 3], eps_tol=1e-6)
    assert len(lad.rungs) == 2 and "cloud said no" in lad.stop_reason


def test_pps_params_respect_bluequbit_floor():
    ps = A.pps_params()
    assert ps[0] == A.PPS_DELTA0 and min(ps) >= A.BQ_MIN_PPS_THRESHOLD
    assert ps[1] / ps[0] == pytest.approx(A.PPS_RATIO)


def test_predict_next_power_law_and_distrust_of_flat_slope():
    assert A.predict_next([16, 32, 64], [1.0, 2.0, 4.0], 128, inverse=False) == pytest.approx(8.0)
    # warm-up dominated: the first rung looks slower than the second -> slope < 1 -> 2 x last
    assert A.predict_next([16, 32], [3.0, 1.0], 64, inverse=False) == pytest.approx(2.0)
    assert A.predict_next([16], [1.0], 32, inverse=False) is None
    assert A.predict_next([1e-3, 5e-4], [1.0, 2.0], None, inverse=True) is None


# ── fusion ───────────────────────────────────────────────────────────────────
def _lad(method, rungs, conv, degen=None):
    l = A.Ladder(method)
    for k, v in enumerate(rungs):
        l.rungs.append(A.Rung(method, None, v, 0.1, degenerate=(degen[k] if degen else [False] * len(v))))
    l.converged = conv
    return l


def test_fuse_priorities_agreement_and_disagreement():
    pps = _lad("pps", [[0.50, 0.10, 0.0], [0.51, 0.10, 0.0]], [True, True, False])
    mps = _lad("mps", [[0.52, 0.61, 0.4], [0.52, 0.60, 0.3]], [True, True, False])
    lc = A.Rung("lightcone", None, [None, None, 0.25], 0.0)
    est = A.fuse(3, [pps, mps], lightcone=lc)
    assert est[2].basis == "lightcone" and est[2].value == 0.25
    assert est[0].basis == "agree" and 0.51 <= est[0].value <= 0.52
    assert est[1].basis.startswith("disagree") and est[1].value == pytest.approx(0.10)
    assert est[1].sigma == pytest.approx(0.5)


def test_fuse_wide_method_does_not_override_tight_one():
    est = A.fuse(1, [_lad("pps", [[0.10], [0.10]], [True]), _lad("mps", [[0.80], [0.60]], [True])])
    assert est[0].basis == "agree" and est[0].value == pytest.approx(0.10, abs=1e-6)


def test_fuse_degenerate_zero_cannot_beat_mps():
    """The reviewer's false-convergence case: PPS truncated to an empty operator twice."""
    pps = _lad("pps", [[0.0], [0.0]], [True], degen=[[True], [True]])
    mps = _lad("mps", [[0.41], [0.40]], [True])
    est = A.fuse(1, [pps, mps])
    assert est[0].value == pytest.approx(0.40) and est[0].basis == "mps"


def test_fuse_unconverged_is_flagged_not_hidden():
    est = A.fuse(1, [_lad("pps", [[0.4], [0.2]], [False])])
    assert est[0].basis == "unconverged:pps" and est[0].sigma == pytest.approx(0.2)


def test_fuse_refuses_when_nothing_estimated():
    with pytest.raises(ValueError):
        A.fuse(1, [])


# ── scoring and planning ─────────────────────────────────────────────────────
def test_score_perfect_noisy_and_constant():
    x = [0.1, 0.5, -0.3, 0.8]
    s = A.score(x, x)
    assert s["pearson"] == pytest.approx(1.0) and s["snr"] == math.inf
    s2 = A.score([0.2, 0.4, -0.2, 0.7], x)
    assert 0.9 < s2["pearson"] < 1.0 and s2["rmse"] == pytest.approx(0.1)
    assert math.isnan(A.score([0.0] * 4, x)["pearson"])


def test_plan_prefers_uncertain_cheap_shrinking_work():
    acts = A.plan([("c1", "pps", 1e-3, 10.0, [0.2, 0.2], [0.5, 0.5]),
                   ("c2", "mps", 64, 10.0, [0.01, 0.01], [0.5, 0.5]),
                   ("c3", "pps", 1e-3, 1.0, [0.2, 0.2], [0.5, 0.5]),
                   ("c4", "mps", 64, 1.0, [0.2, 0.2], [1.0, 1.0])],   # stuck: no gain
                  all_values=[0.0, 0.5, 1.0])
    assert [a.circuit for a in acts[:3]] == ["c3", "c1", "c2"]
    assert acts[-1].circuit == "c4" and acts[-1].gain == 0.0


def test_plan_handles_infinite_sigmas():
    acts = A.plan([("c1", "pps", 1e-3, 1.0, [math.inf, 0.1], [0.5, 0.5])], all_values=[0.0, 1.0])
    assert math.isfinite(acts[0].priority) and acts[0].gain > 0


# ── the campaign ─────────────────────────────────────────────────────────────
def test_campaign_settles_both_and_stops_when_nothing_is_left_to_buy():
    easy, _ = _ki(n=6, steps=1, th=0.3)
    e2 = [(i, i + 1) for i in range(9)]
    hard = A.kicked_ising(10, 6, 1.0, e2)
    targets = {"easy": (easy, [A.parse_observable(s, 6) for s in ["Z0", "Z3", "X0Z1Y2"]]),
               "hard": (hard, [A.parse_observable(s, 10) for s in ["Z0", "Z5", "Z3Z4", "X0Z1Y2"]])}
    rep = A.Campaign(targets, A.local_engines(), budget=60, log_path=None, lightcone=False).run()
    assert rep["spent"] < 60          # both methods cross-checked, and it stopped before the budget
    for cid, (qc, ops) in targets.items():
        ex = A.ExactEngine().run(qc, ops).values
        got = [e["value"] for e in rep["circuits"][cid]["estimates"]]
        assert np.abs(np.array(got) - ex).max() < 1e-2


def test_campaign_respects_a_tiny_budget():
    qc, _ = _ki(n=10, steps=6, th=0.9)
    ops = [A.parse_observable(s, 10) for s in ["Z0", "Z5"]]
    rep = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=0.5, first_rung_guess=0.01,
                     log_path=None, lightcone=False).run()
    assert rep["steps"] <= 4


def test_campaign_light_cone_covers_and_skips_ladders():
    qc, _ = _ki(n=8, steps=2, th=0.5)
    ops = [A.parse_observable(s, 8) for s in ["Z0", "X0Z1Y2"]]
    rep = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=30, log_path=None).run()
    assert rep["steps"] == 0
    assert {e["basis"] for e in rep["circuits"]["c"]["estimates"]} == {"lightcone"}


def test_estimate_partial_light_cone_coverage(monkeypatch):
    """One observable fits the cone, one does not: the ladders run on the second only and the
    results are re-indexed correctly."""
    qc, _ = _ki(n=10, steps=4, th=0.8)
    ops = [A.parse_observable("Z0", 10), A.parse_observable("Z0Z9", 10)]
    monkeypatch.setattr(A, "LIGHTCONE_MAX_QUBITS", 7)
    ex = A.ExactEngine().run(qc, ops).values
    res = A.estimate(qc, ops, budget_s=30, log_path=None)
    bases = [e["basis"] for e in res["estimates"]]
    assert bases[0] == "lightcone" and bases[1] != "lightcone"
    assert np.abs(np.array([e["value"] for e in res["estimates"]]) - ex).max() < 2e-2


def test_estimate_writes_run_log(tmp_path):
    qc, _ = _ki(n=6, steps=2)
    log = tmp_path / "runs.jsonl"
    A.estimate(qc, [A.parse_observable("Z0", 6)], budget_s=10, log_path=log, label="t",
               use_lightcone=False)
    rows = log.read_text().splitlines()
    assert rows and '"circuit": "t"' in rows[0]


def test_pulse_passes():
    r = A.pulse()
    assert r["success"], r["message"]


# ── review round (2026-10-14): each fix shown to fire ──────────────────
def test_report_never_crashes_and_counts_missing_estimates():
    qc, _ = _ki(n=8, steps=3)
    ops = [A.parse_observable("Z0", 8), A.parse_observable("Z1", 8)]
    rep = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=0.0, first_rung_guess=1.0,
                     log_path=None, lightcone=False).run()
    assert rep["steps"] == 0 and rep["missing_estimates"] == 2
    assert {e["basis"] for e in rep["circuits"]["c"]["estimates"]} == {"none"}


def test_credit_budget_refused_for_local_engines():
    with pytest.raises(ValueError, match="cloud"):
        A.Campaign({}, A.local_engines(), budget=10, cost_metric="usd", log_path=None)
    with pytest.raises(ValueError):
        A.Campaign({}, A.local_engines(), budget=10, cost_metric="minutes", log_path=None)


class _Flaky(A.Engine):
    method = "flaky"

    def __init__(self, exc, fail_times):
        self.exc, self.fail_times, self.calls = exc, fail_times, 0

    def run(self, target, ops, param):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return A.Rung(self.method, param, [0.25] * len(ops), 0.05, degenerate=[False] * len(ops))


def test_transient_failure_is_retried_and_refusal_is_not():
    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4)]
    flaky = _Flaky(ConnectionError("queue hiccup"), 1)
    camp = A.Campaign({"c": (qc, ops)}, [(flaky, [1, 2], False)], budget=10, log_path=None,
                      lightcone=False, backoff_s=0.0)
    rep = camp.run()
    assert flaky.calls >= 2 and not rep["failures"] and rep["circuits"]["c"]["estimates"][0]["value"] == 0.25
    refused = _Flaky(ValueError("bad option"), 99)
    rep = A.Campaign({"c": (qc, ops)}, [(refused, [1, 2], False)], budget=10, log_path=None,
                     lightcone=False, backoff_s=0.0).run()
    assert refused.calls == 1 and "bad option" in rep["failures"][0]["error"]


def test_deadline_stops_new_rungs():
    qc, _ = _ki(n=8, steps=4)
    ops = [A.parse_observable("Z0", 8)]
    rep = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=100, log_path=None,
                     lightcone=False).run(deadline_s=0.0)
    assert rep["steps"] == 0


def test_empty_parameter_list_is_handled():
    qc, _ = _ki(n=4, steps=1)
    rep = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(A.MPSEngine(), [], False)],
                     budget=5, log_path=None, lightcone=False).run()
    assert rep["missing_estimates"] == 1


def test_bluequbit_engine_strips_final_measurements_before_sending():
    seen = []

    class _C(_Client):
        def run(self, target, **kw):
            seen.append(target)
            return super().run(target, **kw)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()
    A.BlueQubitEngine(_C([0.0]), "mps.cpu").run(qc, [A.parse_observable("Z0", 2)], 16)
    assert not any(ins.operation.name == "measure" for ins in seen[0].data)


def test_sole_degenerate_value_is_labelled_a_guess():
    est = A.fuse(1, [_lad("pps", [[0.0], [0.0]], [False], degen=[[True], [True]])])
    assert est[0].basis == "degenerate:pps" and est[0].sigma == 1.0


def test_per_observable_fallback_keeps_true_indices():
    qc, _ = _ki(n=6, steps=2)
    ops = [A.parse_observable("Z0", 6), A.parse_observable("Z1", 6)]
    camp = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=10, log_path=None, lightcone=False)
    lad = A.Ladder("mps")
    lad.rungs.append(A.Rung("mps", 16, [math.nan, 0.3], 0.1, degenerate=[True, False]))
    camp.tracks[1].ladder = lad
    est = camp.estimates("c", strict=False)
    assert est[1] is not None and est[1].index == 1


# ── review round 2 (2026-10-14) ────────────────────────────────────────
def test_parse_leading_coefficient_and_identity():
    d = dict((lbl, c.real) for lbl, c in A.parse_observable("0.5Z0-2X1+1.5*I", 2).to_list())
    assert d == {"IZ": 0.5, "XI": -2.0, "II": 1.5}
    qc, _ = _ki(n=2, steps=1)
    v = A.ExactEngine().run(qc, [A.parse_observable("2*I+Z0", 2)]).values[0]
    assert v == pytest.approx(2.0 + A.ExactEngine().run(qc, [A.parse_observable("Z0", 2)]).values[0])


def test_pps_refuses_non_hermitian_observable():
    from qiskit.quantum_info import SparsePauliOp

    qc, _ = _ki(n=3, steps=1)
    with pytest.raises(ValueError, match="Hermitian"):
        A.PPSEngine().run(qc, [SparsePauliOp(["IIZ"], coeffs=[1j])], 1e-3)


def test_one_converged_method_does_not_settle_an_observable():
    """DeepSeek's finding: a single converged method must leave work for the second one."""
    qc, _ = _ki(n=6, steps=2)
    ops = [A.parse_observable("Z0", 6)]
    camp = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=10, log_path=None, lightcone=False)
    camp.tracks[0].ladder = _lad("pps", [[0.3], [0.3]], [True])
    assert camp._sigmas("c")[0] > 0
    camp.tracks[1].ladder = _lad("mps", [[0.3], [0.3]], [True])
    assert camp._sigmas("c")[0] == 0.0          # two methods agreeing do settle it


def test_engine_returning_wrong_count_is_a_reported_failure():
    class _Short(A.Engine):
        method = "short"

        def run(self, target, ops, param):
            return A.Rung(self.method, param, [0.1], 0.01, degenerate=[False])

    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4), A.parse_observable("Z1", 4)]
    rep = A.Campaign({"c": (qc, ops)}, [(_Short(), [1], False)], budget=5, log_path=None,
                     lightcone=False).run()
    assert "returned 1 values for 2" in rep["failures"][0]["error"]


# ── the day: cloud campaign through a simulated BlueQubit ────────────────────
class _SimBQ:
    """Answers like BlueQubit: real local engines behind the SDK's call shape, the SDK's own
    pauli-path validation, a credit charge per job, async jobs collected by wait()."""

    def __init__(self, price=0.01):
        self.price, self.jobs, self.spent, self.n = price, {}, 0.0, 0

    def _value(self, target, device, ps, options):
        from qiskit.quantum_info import SparsePauliOp

        op = SparsePauliOp.from_list([(l, c) for l, c in ps])
        if device == "pauli-path":
            return A.PPSEngine().run(target, [op], options["pauli_path_truncation_threshold"]).values[0]
        return A.MPSEngine().run(target, [op], options["mps_bond_dimension"]).values[0]

    def run(self, target, device, pauli_sum=None, options=None, job_name=None, asynchronous=False):
        from bluequbit import BQClient

        if device == "pauli-path":
            BQClient._validate_pps_job(options, target, pauli_sum)
            ev = self._value(target, device, pauli_sum, options)
        else:
            ev = [self._value(target, device, ps, options) for ps in pauli_sum]
        self.n += 1
        self.spent += self.price
        res = _Res(ev, cost=self.price, job_id=f"J{self.n}")
        self.jobs[res.job_id] = res
        return res

    def wait(self, job_ids, timeout=None):
        return [self.jobs[j] for j in job_ids]

    def cancel(self, job_id):
        pass


def _day_spec(tmp_path, budget=1.0, deadline=None):
    qc, _ = _ki(n=8, steps=3, th=0.8)
    p = tmp_path / "c1.qasm"
    from qiskit import qasm2
    p.write_text(qasm2.dumps(qc))
    spec = {"budget_usd": budget, "circuits": {
        "c1": {"target": str(p), "n_qubits": 8, "observables": ["Z0", "Z3Z4", "X0Z1Y2"]}}}
    if deadline:
        spec["deadline_utc"] = deadline
    f = tmp_path / "spec.json"
    f.write_text(__import__("json").dumps(spec))
    return f, qc


def test_run_day_end_to_end_through_simulated_bluequbit(tmp_path):
    f, qc = _day_spec(tmp_path)
    bq = _SimBQ()
    rep = A.run_day(A.load_day_spec(f), bq, out_dir=tmp_path / "day")
    assert bq.spent <= 1.0 + 0.05 and rep["spent"] == pytest.approx(bq.spent)
    ex = A.ExactEngine().run(qc, [A.parse_observable(s, 8) for s in ["Z0", "Z3Z4", "X0Z1Y2"]]).values
    got = [e["value"] for e in rep["circuits"]["c1"]["estimates"]]
    assert np.abs(np.array(got) - ex).max() < 1e-2
    csv = Path(rep["submission_path"]).read_text().splitlines()
    assert csv[0] == "circuit,index,observable,value,sigma,basis" and len(csv) == 4


def test_run_day_past_deadline_runs_nothing_and_writes_zeros_flagged(tmp_path):
    f, _ = _day_spec(tmp_path, deadline="2000-01-01T00:00:00+00:00")
    bq = _SimBQ()
    rep = A.run_day(A.load_day_spec(f), bq, out_dir=tmp_path / "day")
    assert bq.n == 0 and len(rep["missing"]) == 3
    assert ",0," in Path(rep["submission_path"]).read_text().splitlines()[1]


def test_day_spec_validation_and_spend_confirmation(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"circuits": {}}')
    with pytest.raises(ValueError):
        A.load_day_spec(bad)
    bad.write_text('{"budget_usd": 1, "circuits": {"x": {"target": "x"}}}')
    with pytest.raises(ValueError, match="observables"):
        A.load_day_spec(bad)
    f, _ = _day_spec(tmp_path)
    assert A._cli(["day", "--spec", str(f)]) == 2          # refuses without --confirm-spend


def test_day_qasm_qubit_count_must_match_spec(tmp_path):
    f, _ = _day_spec(tmp_path)
    spec = A.load_day_spec(f)
    spec["circuits"]["c1"]["n_qubits"] = 9
    with pytest.raises(ValueError, match="qubits"):
        A.day_targets(spec)


# ── review round 3 (recorded panel 759217dc833d) ───────────────────────
def test_cloud_cpu_is_not_called_exact_and_unknown_device_refused():
    assert A.BlueQubitEngine(_Client([0]), "cpu").method == "statevector"
    with pytest.raises(ValueError):
        A.BlueQubitEngine(_Client([0]), "quantum-ish")


def test_cloud_non_finite_value_refused():
    with pytest.raises(RuntimeError, match="non-finite"):
        A.BlueQubitEngine(_Client([float("nan")]), "mps.cpu").run(QuantumCircuit(1), [A.parse_observable("Z0", 1)], 16)


def test_negative_budgets_refused(tmp_path):
    with pytest.raises(ValueError):
        A.Campaign({}, A.local_engines(), budget=-1, log_path=None)
    bad = tmp_path / "b.json"
    bad.write_text('{"budget_usd": -5, "circuits": {"x": {"target": "x", "n_qubits": 1, "observables": ["Z0"]}}}')
    with pytest.raises(ValueError):
        A.load_day_spec(bad)


def test_submission_csv_quotes_observables_with_commas(tmp_path):
    import csv as _csv

    f, _ = _day_spec(tmp_path)
    spec = A.load_day_spec(f)
    spec["circuits"]["c1"]["observables"][0] = "0.5*Z0+Z1"
    rep = A.run_day(spec, _SimBQ(), out_dir=tmp_path / "day")
    with Path(rep["submission_path"]).open(newline="") as fh:
        rows = list(_csv.reader(fh))
    assert all(len(r) == 6 for r in rows) and rows[1][2] == "0.5*Z0+Z1"


def test_one_point_prediction_is_cubic_not_double():
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(A.MPSEngine(), [16, 32], False)],
                      budget=100, log_path=None, lightcone=False)
    tr = camp.tracks[0]
    tr.ladder.rungs.append(A.Rung("mps", 16, [0.1], 1.0, degenerate=[False]))
    tr.pos = 1
    assert camp._predict(tr) == pytest.approx(8.0)


def test_exact_engine_refuses_non_hermitian():
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="Hermitian"):
        A.ExactEngine().run(qc, [SparsePauliOp(["X"], coeffs=[1j])])


# ── final panel round (50281bbbd8c1) ─────────────────────────────────────────
def test_settled_observables_are_not_rerun_and_stay_settled():
    """Once two methods agree on an observable, later rungs skip it (it costs a cloud job)
    and it keeps its value."""
    calls = []

    class _Count(A.PPSEngine):
        def run(self, target, ops, param):
            calls.append(len(ops))
            return super().run(target, ops, param)

    qc, edges = _ki(n=10, steps=6, th=0.9)
    ops = [A.parse_observable(s, 10) for s in A.local_observables(10, edges)]
    camp = A.Campaign({"c": (qc, ops)}, [(_Count(), A.pps_params(), True),
                                        (A.MPSEngine(), A.MPS_CHIS, False)],
                      budget=60, log_path=None, lightcone=False)
    rep = camp.run()
    assert calls[0] == len(ops) and min(calls) < len(ops)
    ex = A.ExactEngine().run(qc, ops).values
    got = [e["value"] for e in rep["circuits"]["c"]["estimates"]]
    assert np.abs(np.array(got) - ex).max() < 1e-2


def test_cloud_wait_capped_by_deadline():
    seen = {}

    class _W(_Client):
        def wait(self, job_ids, timeout=None):
            seen["timeout"] = timeout
            return super().wait(job_ids, timeout)

    qc, _ = _ki(n=4, steps=1)
    eng = A.BlueQubitEngine(_W([0.2]), "pauli-path")
    A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(eng, [1e-3], True)], budget=10,
               cost_metric="usd", log_path=None, lightcone=False).run(deadline_s=50)
    assert seen["timeout"] is not None and seen["timeout"] <= 50


def test_cloud_result_shapes_array_and_dict():
    class _Arr(_Client):
        def wait(self, job_ids, timeout=None):
            return {j: _Res(np.array([0.1, 0.2]), job_id=j) for j in job_ids}

    r = A.BlueQubitEngine(_Arr([0]), "mps.cpu").run(
        QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 16)
    assert r.values == pytest.approx([0.1, 0.2])


def test_pauli_path_results_realigned_to_submission_order():
    class _Shuffled(_Client):
        def wait(self, job_ids, timeout=None):
            return [self.jobs[j] for j in reversed(job_ids)]

    r = A.BlueQubitEngine(_Shuffled([0.1, 0.2, 0.3]), "pauli-path").run(
        QuantumCircuit(3), [A.parse_observable(f"Z{q}", 3) for q in range(3)], 1e-3)
    assert r.values == [0.1, 0.2, 0.3]


def test_failed_wait_cancels_jobs_and_is_not_resubmitted_past_deadline():
    class _Stuck(_Client):
        def wait(self, job_ids, timeout=None):
            raise RuntimeError("queue stuck")

    cl = _Stuck([0.1, 0.2])
    eng = A.BlueQubitEngine(cl, "pauli-path")
    with pytest.raises(RuntimeError, match="cancel requested for 2 of 2"):
        eng.run(QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 1e-3)
    assert cl.cancelled == ["J1", "J2"]
    cl2 = _Stuck([0.1])
    qc, _ = _ki(n=4, steps=1)
    rep = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])},
                     [(A.BlueQubitEngine(cl2, "pauli-path"), [1e-3], True)], budget=1.0,
                     cost_metric="usd", log_path=None, lightcone=False, backoff_s=30.0).run(deadline_s=5)
    assert cl2.k == 1 and rep["failures"]          # backoff would cross the deadline: no resubmit


def test_mps_cloud_wait_is_capped_too():
    seen = {}

    class _W(_Client):
        def wait(self, job_ids, timeout=None):
            seen["timeout"] = timeout
            return super().wait(job_ids, timeout)

    qc, _ = _ki(n=4, steps=1)
    eng = A.BlueQubitEngine(_W([0.2]), "mps.cpu")
    A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(eng, [16], False)], budget=10,
               cost_metric="usd", log_path=None, lightcone=False).run(deadline_s=50)
    assert seen["timeout"] is not None and seen["timeout"] <= 50


def test_two_agreeing_methods_outvote_a_tight_outlier():
    a = _lad("mps", [[0.40], [0.40]], [True])
    b = _lad("mps2", [[0.41], [0.405]], [True])
    c = _lad("pps", [[0.90], [0.90]], [True])          # tightest (sigma floor) but alone
    est = A.fuse(1, [a, b, c])
    assert est[0].value == pytest.approx(0.40, abs=0.01)


def test_score_refuses_length_mismatch():
    with pytest.raises(ValueError):
        A.score([0.1, 0.2], [0.1, 0.2, 0.3])


def test_day_spec_refuses_empty_observables_and_naive_deadline(tmp_path):
    f = tmp_path / "s.json"
    f.write_text('{"budget_usd": 1, "circuits": {"x": {"target": "x", "n_qubits": 1, "observables": []}}}')
    with pytest.raises(ValueError, match="no observables"):
        A.load_day_spec(f)
    f.write_text('{"budget_usd": 1, "deadline_utc": "2026-10-25T15:00:00", '
                 '"circuits": {"x": {"target": "x", "n_qubits": 1, "observables": ["Z0"]}}}')
    with pytest.raises(ValueError, match="timezone"):
        A.load_day_spec(f)


def test_prediction_overflow_is_infinite_not_a_crash():
    assert A.predict_next([16, 32], [1.0, 1e300], 1e6, inverse=False) == math.inf


def test_memory_error_is_not_retried():
    flaky = _Flaky(MemoryError(), 99)
    qc, _ = _ki(n=4, steps=1)
    A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(flaky, [1, 2], False)], budget=10,
               log_path=None, lightcone=False, backoff_s=0.0).run()
    assert flaky.calls == 1


def test_calibrate_trusts_the_accurate_method_and_fuse_uses_it():
    class _Biased(A.MPSEngine):
        method = "mps"

        def run(self, target, ops, param):
            r = super().run(target, ops, param)
            r.values = [v + 0.2 for v in r.values]
            return r

    qc, edges = _ki(n=8, steps=3, th=0.8)
    ops = [A.parse_observable(s, 8) for s in A.local_observables(8, edges)]
    ex = A.ExactEngine().run(qc, ops).values
    cal = A.calibrate([(qc, ops, ex)], engines=[(A.PPSEngine(), A.pps_params(), True),
                                                (_Biased(), A.MPS_CHIS, False)], budget_s=30)
    assert cal["trust"]["pps"] == 1.0 and cal["trust"]["mps"] < 0.2
    pps = _lad("pps", [[0.10], [0.10], [0.10]], [True])
    mps = _lad("mps", [[0.50], [0.50]], [True])
    assert A.fuse(1, [pps, mps], trust=cal["trust"])[0].value == pytest.approx(0.10, abs=1e-3)


class _Converging(A.Engine):
    """Deterministic stand-in: value approaches `target` as 1/2^rung, each rung declares a cost
    of `cost` seconds. Tests allocation logic, not physics."""

    def __init__(self, method, target, cost):
        self.method, self.target, self.cost = method, target, cost

    def run(self, target, ops, param):
        k = int(param) - 19          # params 20, 21, ... grow ~5% a rung, so the cubic
        return A.Rung(self.method, param, [self.target + 0.5 / 2 ** k for _ in ops], self.cost,
                      degenerate=[False] * len(ops))   # one-point guard predicts ~1.16x


def _fake_engines(cost_pps=1.0, cost_mps=1.0):
    return [(_Converging("pps", 0.30, cost_pps), list(range(20, 40)), False),
            (_Converging("mps", 0.31, cost_mps), list(range(20, 40)), False)]


def test_slow_converging_circuit_gets_more_rungs():
    class _Rate(A.Engine):
        def __init__(self, method, rate):
            self.method, self.rate = method, rate

        def run(self, target, ops, param):
            return A.Rung(self.method, param, [0.3 + 0.5 * self.rate ** (int(param) - 19) for _ in ops], 1.0,
                          degenerate=[False] * len(ops))

    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4)]
    eng = [(_Rate("pps", 0.01), list(range(20, 40)), False), (_Rate("mps", 0.01), list(range(20, 40)), False)]
    slow = [(_Rate("pps", 0.7), list(range(20, 40)), False), (_Rate("mps", 0.7), list(range(20, 40)), False)]
    camp = A.Campaign({"fast": (qc, ops)}, eng, budget=20, log_path=None, lightcone=False)
    camp.targets["slow"] = (qc, ops)
    camp.tracks += [A._Track("slow", e, list(p), inv, A.Ladder(e.method)) for e, p, inv in slow]
    rep = camp.run()
    rungs = {c: sum(l["rungs"] for l in v["ladders"]) for c, v in rep["circuits"].items()}
    assert rungs["slow"] > rungs["fast"]


def test_trust_steers_spending_not_only_fusion():
    """With MPS distrusted, the campaign gives Pauli-path more rungs and MPS fewer."""
    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4), A.parse_observable("Z1", 4)]

    def rungs(trust):
        rep = A.Campaign({"c": (qc, ops)}, _fake_engines(), budget=9, log_path=None,
                         lightcone=False, trust=trust, eps_tol=1e-4).run()
        return {l["method"]: l["rungs"] for l in rep["circuits"]["c"]["ladders"]}

    plain, steered = rungs(None), rungs({"pps": 1.0, "mps": 0.05})
    assert steered["pps"] > plain["pps"] and steered["mps"] < plain["mps"]


# ── second review round on the committed build ─────────────────────────────
def test_failed_submission_cancels_jobs_already_submitted():
    class _Half(_Client):
        def run(self, target, **kw):
            if self.k == 1:
                raise ConnectionError("dropped mid-batch")
            return super().run(target, **kw)

    cl = _Half([0.1, 0.2, 0.3])
    with pytest.raises(ConnectionError):
        A.BlueQubitEngine(cl, "pauli-path").run(
            QuantumCircuit(3), [A.parse_observable(f"Z{q}", 3) for q in range(3)], 1e-3)
    assert cl.cancelled == ["J1"]


def test_results_without_job_ids_are_refused_not_trusted_by_order():
    class _NoIds(_Client):
        def wait(self, job_ids, timeout=None):
            return [_Res(0.1, job_id=None) for _ in job_ids]

    with pytest.raises(RuntimeError, match="returned results"):
        A.BlueQubitEngine(_NoIds([0.1]), "pauli-path").run(
            QuantumCircuit(1), [A.parse_observable("Z0", 1)], 1e-3)


def test_cloud_true_zero_settles_on_one_witness_and_stops_buying():
    """A symmetry zero: cloud pauli-path returns exact 0.0 (flagged degenerate) every rung,
    MPS converges on 0.0. No second witness can ever appear, so it must count as settled."""
    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4)]
    camp = A.Campaign({"c": (qc, ops)}, A.local_engines(), budget=10, log_path=None, lightcone=False)
    camp.tracks[0].ladder = _lad("pps", [[0.0], [0.0]], [False], degen=[[True], [True]])
    camp.tracks[1].ladder = _lad("mps", [[0.0], [0.0]], [True])
    assert camp._sigmas("c")[0] == 0.0
    camp.tracks[0].ladder = _lad("pps", [[0.0]], [False], degen=[[True]])   # one rung: not yet
    assert camp._sigmas("c")[0] > 0


def test_campaign_parameters_validated():
    for kw in ({"agree": 0}, {"eps_tol": -1}, {"retries": -1}, {"backoff_s": -1}):
        with pytest.raises(ValueError):
            A.Campaign({}, A.local_engines(), budget=1, log_path=None, **kw)


def test_no_cloud_rung_starts_with_under_a_second_left():
    cl = _Client([0.2])
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])},
                      [(A.BlueQubitEngine(cl, "pauli-path"), [1e-3], True)], budget=10,
                      cost_metric="usd", log_path=None, lightcone=False)
    camp._stop_at = __import__("time").monotonic() + 0.5
    camp._run(camp.tracks[0])
    assert cl.k == 0 and camp.tracks[0].ladder.stop_reason == "deadline reached"


def test_score_nan_estimate_is_not_infinite_snr():
    s = A.score([0.1, math.nan, 0.3], [0.1, 0.2, 0.3])
    assert math.isnan(s["snr"])


def test_pps_params_never_below_bluequbit_floor_even_at_the_edge():
    """A step a hair under the floor (1e-5 x (1 - 1e-10)) was let through by the old relative
    tolerance and would have been refused by BlueQubit's options check on the day."""
    ps = A.pps_params(delta0=2e-5, ratio=0.5 * (1 - 1e-10), floor=1e-5)
    assert ps == [2e-5] and min(ps) >= 1e-5


def test_day_refuses_both_mps_devices_and_report_file_is_complete(tmp_path):
    f, _ = _day_spec(tmp_path)
    spec = A.load_day_spec(f)
    with pytest.raises(ValueError, match=r"mps\.cpu or mps\.gpu"):
        A.run_day({**spec, "devices": ["mps.cpu", "mps.gpu"]}, _SimBQ(), out_dir=tmp_path / "d1")
    rep = A.run_day(spec, _SimBQ(), out_dir=tmp_path / "d2")
    saved = __import__("json").loads(Path(rep["report_path"]).read_text())
    assert {"report_path", "submission_path", "missing"} <= set(saved)


# ── resume, checkpoint and the overnight watchdog (2026-10-17) ───────────────
def _long_spec(tmp_path, budget=1.0, deadline="2099-01-01T00:00:00+00:00"):
    f, qc = _day_spec(tmp_path, budget=budget, deadline=deadline)
    return f, qc


def test_resume_buys_nothing_twice(tmp_path):
    f, _ = _long_spec(tmp_path, budget=0.05)                 # small: stops partway
    out = tmp_path / "day"
    first = _SimBQ()
    A.run_day(A.load_day_spec(f), first, out_dir=out)
    logged = len((out / "runs.jsonl").read_text().splitlines())
    spec = A.load_day_spec(f)
    spec["budget_usd"] = 1.0                                 # more credit, same run
    second = _SimBQ()
    rep = A.run_day(spec, second, out_dir=out, resume=True)
    assert logged > 0 and rep["restored_rungs"] == logged
    rows = [__import__("json").loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    seen = [(r["circuit"], r["method"], r["param"]) for r in rows]
    assert len(seen) == len(set(seen)), "a rung was bought twice"
    assert rep["spent"] == pytest.approx(first.spent + second.spent)


def test_heartbeat_and_latest_submission_written_every_rung(tmp_path):
    """Between consecutive cloud calls the heartbeat's step count must have moved: the
    checkpoint runs after every rung, not only at the end."""
    import json as _j
    f, _ = _long_spec(tmp_path)
    out = tmp_path / "day"
    steps_seen = []

    class _Peek(_SimBQ):
        def run(self, *a, **kw):
            if (out / "heartbeat.json").exists():
                steps_seen.append(_j.loads((out / "heartbeat.json").read_text())["steps"])
            return super().run(*a, **kw)

    rep = A.run_day(A.load_day_spec(f), _Peek(), out_dir=out)
    hb = _j.loads((out / "heartbeat.json").read_text())
    assert hb["done"] is True and hb["steps"] == rep["steps"]
    distinct = sorted(set(steps_seen))
    assert distinct == list(range(len(distinct))) and len(distinct) >= rep["steps"] - 1
    assert (out / "submission_latest.csv").read_text().startswith("circuit,index")


def test_failing_checkpoint_never_stops_the_run():
    qc, _ = _ki(n=4, steps=1)

    def boom(c):
        raise OSError("disk full")

    rep = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, _fake_engines(), budget=6,
                     log_path=None, lightcone=False, on_step=boom).run()
    assert rep["steps"] > 1 and any("checkpoint failed" in f["error"] for f in rep["failures"])


class _Mail:
    def __init__(self):
        self.sent = []

    def __call__(self, subject, body):
        self.sent.append(subject)
        return True, "ok"


def _armed(tmp_path, deadline="2099-01-01T00:00:00+00:00", hb=None):
    import json as _j
    out = tmp_path / "day"
    out.mkdir(exist_ok=True)
    af = tmp_path / "ARMED.json"
    af.write_text(_j.dumps({"spec": "s.json", "out_dir": str(out), "deadline_utc": deadline,
                            "budget_usd": 10.0}))
    if hb is not None:
        (out / "heartbeat.json").write_text(_j.dumps(hb))
    return af, out


def test_watch_idle_when_not_armed(tmp_path):
    spawned, mail = [], _Mail()
    r = A.watch(armed_file=tmp_path / "none.json", spawn=lambda s, o: spawned.append(s) or 1, send=mail)
    assert r["state"] == "idle" and not spawned and not mail.sent


def test_watch_restarts_dead_run_then_gives_up_without_progress(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 999999, "ts": 0, "steps": 4, "spent": 1, "budget": 10})
    spawned, mail = [], _Mail()
    states = [A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: False,
                      spawn=lambda s, o: spawned.append(s) or 4242, send=mail)["state"]
              for _ in range(4)]
    assert states == ["restarted", "restarted", "restarted", "gave_up"]
    assert len(spawned) == 3 and not af.exists()
    assert "[Augur] watchdog gave up" in mail.sent


def test_watch_progress_resets_the_restart_count(tmp_path):
    import json as _j
    af, out = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 4, "spent": 1, "budget": 10})
    kw = dict(armed_file=af, now=lambda: 1e9, alive=lambda p: False, spawn=lambda s, o: 7, send=_Mail())
    A.watch(**kw)
    A.watch(**kw)
    (out / "heartbeat.json").write_text(_j.dumps({"pid": 1, "ts": 0, "steps": 9, "spent": 2, "budget": 10}))
    A.watch(**kw)
    st = _j.loads((out / "watch_state.json").read_text())
    assert st["restarts_without_progress"] == 1


def test_watch_never_restarts_a_live_run_even_when_quiet(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 4, "spent": 1, "budget": 10})
    spawned, mail = [], _Mail()
    r = A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: True,
                spawn=lambda s, o: spawned.append(s) or 1, send=mail)
    assert r["state"] == "quiet" and not spawned and mail.sent == ["[Augur] run quiet"]


def test_watch_live_beating_run_is_left_alone(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 1e9, "steps": 4, "spent": 1, "budget": 10})
    spawned, mail = [], _Mail()
    r = A.watch(armed_file=af, now=lambda: 1e9 + 10, alive=lambda p: True,
                spawn=lambda s, o: spawned.append(s) or 1, send=mail)
    assert r["state"] == "running" and not spawned and not mail.sent


def test_watch_finished_and_deadline_disarm_with_one_final_email(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 9, "done": True, "spent": 3,
                                 "budget": 10, "submission_path": "x.csv", "missing": 0})
    mail = _Mail()
    assert A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: False, spawn=lambda s, o: 1,
                   send=mail)["state"] == "done"
    assert mail.sent == ["[Augur] run finished"] and not af.exists()
    (tmp_path / "c").mkdir()
    af2, _ = _armed(tmp_path / "c", deadline="2000-01-01T00:00:00+00:00",
                    hb={"pid": 1, "ts": 0, "steps": 1, "spent": 1, "budget": 10})
    mail2 = _Mail()
    assert A.watch(armed_file=af2, now=lambda: 1e9, alive=lambda p: True, spawn=lambda s, o: 1,
                   send=mail2)["state"] == "deadline"
    assert not af2.exists() and mail2.sent == ["[Augur] deadline reached"]


def test_watch_alerts_are_rate_limited(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 1e9, "steps": 4, "spent": 9, "budget": 10})
    mail = _Mail()
    for dt in (0, 60, 120):
        A.watch(armed_file=af, now=lambda d=dt: 1e9 + d, alive=lambda p: True,
                spawn=lambda s, o: 1, send=mail)
    assert mail.sent.count("[Augur] credits at 80%") == 1


def test_arm_needs_a_future_deadline_and_explicit_consent(tmp_path):
    f, _ = _day_spec(tmp_path)                               # no deadline
    with pytest.raises(ValueError, match="deadline"):
        A.arm(f, armed_file=tmp_path / "A.json")
    f2, _ = _day_spec(tmp_path, deadline="2000-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="passed"):
        A.arm(f2, armed_file=tmp_path / "A.json")
    f3, _ = _day_spec(tmp_path, deadline="2099-01-01T00:00:00+00:00")
    assert A._cli(["arm", "--spec", str(f3)]) == 2           # refuses without --confirm-spend
    rec = A.arm(f3, armed_file=tmp_path / "A.json")
    assert rec["deadline_utc"].startswith("2099") and A.disarm(tmp_path / "A.json")


def test_the_watchdogs_restart_command_is_exactly_this_and_parses(tmp_path, monkeypatch):
    """_spawn_day launches day_argv; that argv carries --confirm-spend and --resume and the real
    parser accepts it (parsed only, never dispatched, so nothing is spent)."""
    argv = A.day_argv("spec.json", str(tmp_path / "d"))
    ns = A._parser().parse_args(argv)
    assert ns.cmd == "day" and ns.confirm_spend and ns.resume and str(ns.out_dir) == str(tmp_path / "d")
    seen = {}

    class _P:
        def __init__(self, args, **kw):
            seen["args"] = args
            self.pid = 12345

    monkeypatch.setattr("subprocess.Popen", _P)
    assert A._spawn_day("spec.json", str(tmp_path / "d")) == 12345
    assert seen["args"][1:3] == ["-m", "augur"] and seen["args"][3:] == argv


# ── Family round on the watchdog (db89cfffde4d, 3fd518a35adc) ───────────────
def test_watch_does_not_spawn_twice_while_the_new_run_is_starting(tmp_path):
    """After a restart, until the new run writes its first heartbeat the old dead pid is all
    the heartbeat shows; a second spawn then would run two paid campaigns at once."""
    af, _ = _armed(tmp_path, hb={"pid": 111, "ts": 0, "steps": 4, "spent": 1, "budget": 10})
    spawned = []
    kw = dict(armed_file=af, now=lambda: 1e9, alive=lambda p: p == 4242,
              spawn=lambda s, o: spawned.append(s) or 4242, send=_Mail())
    assert A.watch(**kw)["state"] == "restarted"
    assert A.watch(**kw)["state"] == "starting"
    assert len(spawned) == 1


def test_pid_alive_requires_an_augur_day_process():
    import os
    import subprocess
    import sys as _s
    assert A._pid_alive(os.getpid()) is False                  # pytest: alive, but not Augur
    p = subprocess.Popen([_s.executable, "-c", "import time; time.sleep(5)", "augur", "day"])
    try:
        assert A._pid_alive(p.pid) is True
    finally:
        p.kill()
        p.wait()
    assert A._pid_alive(p.pid) is False


def test_mismatched_reply_cancels_the_paid_jobs():
    class _Wrong(_Client):
        def wait(self, job_ids, timeout=None):
            return [_Res(0.1, job_id="SOMEONE_ELSE")]

    cl = _Wrong([0.1])
    with pytest.raises(RuntimeError, match="cancel requested for 1 of 1"):
        A.BlueQubitEngine(cl, "pauli-path").run(QuantumCircuit(1), [A.parse_observable("Z0", 1)], 1e-3)
    assert cl.cancelled == ["J1"]


def test_interrupt_cancels_the_paid_jobs_and_still_interrupts():
    class _Ctrl(_Client):
        def wait(self, job_ids, timeout=None):
            raise KeyboardInterrupt

    cl = _Ctrl([0.1, 0.2])
    with pytest.raises(KeyboardInterrupt):
        A.BlueQubitEngine(cl, "pauli-path").run(
            QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 1e-3)
    assert cl.cancelled == ["J1", "J2"]


def test_zero_budget_in_heartbeat_is_not_read_as_missing(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 1e9, "steps": 1, "spent": 0, "budget": 0})
    mail = _Mail()
    A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: True, spawn=lambda s, o: 1, send=mail)
    assert "[Augur] credits at 80%" not in mail.sent


# ── SDK 0.19: hidden circuits by id, get_circuit_info, pauli-path.gpu (2026-10-23) ──
class _Info:
    def __init__(self, n):
        self.circuit_id, self.num_qubits = "snr-1-1", n
        self.count_ops, self.depth_1q, self.depth_2q = {"h": 1, "cx": 1}, 1, 1


class _IdClient(_SimBQ):
    """Answers like SDK 0.19 for a stored circuit id: get_circuit_info, and run() with the id
    string resolved server-side to a circuit the test knows."""

    def __init__(self, circuits):
        super().__init__()
        self.circuits, self.info_calls = circuits, []

    def get_circuit_info(self, cid):
        self.info_calls.append(cid)
        return _Info(self.circuits[cid].num_qubits)

    def run(self, target, **kw):
        return super().run(self.circuits[target] if isinstance(target, str) else target, **kw)


def test_day_run_by_id_looks_up_qubit_count_and_records_info(tmp_path):
    qc, _ = _ki(n=6, steps=2)
    spec = {"budget_usd": 1.0, "circuits": {"c1": {"target": "snr-1-1", "observables": ["Z0", "X0Z1Y2"]}}}
    f = tmp_path / "s.json"
    f.write_text(__import__("json").dumps(spec))
    bq = _IdClient({"snr-1-1": qc})
    rep = A.run_day(A.load_day_spec(f), bq, out_dir=tmp_path / "day")
    assert bq.info_calls == ["snr-1-1"]
    assert rep["circuit_info"]["c1"]["num_qubits"] == 6 and rep["circuit_info"]["c1"]["depth_2q"] == 1
    ex = A.ExactEngine().run(qc, [A.parse_observable(s, 6) for s in ["Z0", "X0Z1Y2"]]).values
    got = [e["value"] for e in rep["circuits"]["c1"]["estimates"]]
    assert np.abs(np.array(got) - ex).max() < 1e-2


# prepatch-ok: HEAD required n_qubits for every target; this guards that relaxing it for ids
# (looked up via get_circuit_info) did not also relax it for qasm files, which have no lookup
def test_qasm_target_still_needs_n_qubits(tmp_path):
    f = tmp_path / "s.json"
    f.write_text('{"budget_usd": 1, "circuits": {"c": {"target": "x.qasm", "observables": ["Z0"]}}}')
    with pytest.raises(ValueError, match="n_qubits"):
        A.load_day_spec(f)


def test_pauli_path_gpu_is_one_job_per_observable_and_not_doubled_with_cpu(tmp_path):
    cl = _Client([0.1, -0.2])
    r = A.BlueQubitEngine(cl, "pauli-path.gpu").run(
        QuantumCircuit(2), [A.parse_observable("Z0", 2), A.parse_observable("Z1", 2)], 1e-3)
    assert r.values == [0.1, -0.2] and len(cl.calls) == 2 and r.method == "pps"
    f, _ = _day_spec(tmp_path)
    spec = A.load_day_spec(f)
    with pytest.raises(ValueError, match=r"pauli-path or pauli-path\.gpu"):
        A.run_day({**spec, "devices": ["pauli-path", "pauli-path.gpu"]}, _SimBQ(), out_dir=tmp_path / "d")


# ── Family round on the SDK 0.19 landing (a69f08eae9e6) ──────────────────────
def test_agreeing_group_requires_every_pair_to_agree():
    """a and c each agree with b but not with each other; the group may not hold all three."""
    a = _lad("m1", [[0.40], [0.40]], [True])
    b = _lad("m2", [[0.45], [0.43]], [True])     # sigma 0.02, between a and c
    c = _lad("m3", [[0.46], [0.46]], [True])
    est = A.fuse(1, [a, b, c])
    pair_ab = A.fuse(1, [a, b])[0].value
    pair_bc = A.fuse(1, [b, c])[0].value
    # the result is one mutually agreeing pair, never an average that also pulls in the outlier
    assert est[0].value in (pytest.approx(pair_ab), pytest.approx(pair_bc))


def test_nan_cost_prediction_is_unaffordable(monkeypatch):
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, _fake_engines(), budget=50,
                      log_path=None, lightcone=False)
    monkeypatch.setattr(camp, "_predict", lambda tr: math.nan)
    rep = camp.run()
    assert rep["steps"] == 0


# ── review round on the public code, in five parts (db35825d39a4 .. 46701b87e6a6) ───────
def test_bool_bounce_wait_refused_and_validated_level_is_the_one_stored():
    with pytest.raises(ValueError, match="bounce_wait_s"):
        A.BlueQubitEngine(_Client([0.1]), "pauli-path", bounce_wait_s=True)
    e = A.BlueQubitEngine(_Client([0.1]), "pauli-path", transpilation_level=7,
                          extra_options={"pauli_path_circuit_transpilation_level": 1})
    assert e.transpilation_level == 1


def test_wait_never_gets_a_spent_deadline_and_jobs_are_cancelled():
    class _Rec(_Client):
        def wait(self, job_ids, timeout=None):
            self.timeouts = getattr(self, "timeouts", []) + [timeout]
            return super().wait(job_ids, timeout)

    import time as _t
    cl = _Rec([0.1, 0.2])
    e = A.BlueQubitEngine(cl, "mps.cpu")
    with pytest.raises(RuntimeError, match="deadline passed"):
        e._submit_and_wait(QuantumCircuit(2), [A.parse_observable("Z0", 2)], {}, "n", _t.monotonic() - 1)
    assert not getattr(cl, "timeouts", []) and cl.cancelled == ["J1"]


def test_complex_cloud_expectation_refused_real_complex_accepted():
    ops = [A.parse_observable("Z0", 2)]
    bad = _Client([np.array([0.5 + 0.3j])])
    with pytest.raises(RuntimeError, match="complex expectation"):
        A.BlueQubitEngine(bad, "pauli-path").run(QuantumCircuit(2), ops, 1e-3)
    ok = _Client([np.array([0.5 + 0j])])
    assert A.BlueQubitEngine(ok, "pauli-path").run(QuantumCircuit(2), ops, 1e-3).values == [0.5]


def test_malformed_degenerate_flags_raise_and_empty_means_none():
    assert A._degen(A.Rung("m", 1, [0.1, 0.2], 0.0), 2) == [False, False]
    with pytest.raises(ValueError, match="1 degenerate flags for 2"):
        A._degen(A.Rung("m", 1, [0.1, 0.2], 0.0, degenerate=[True]), 2)


def _bounded_call(expr: str):
    """Run `expr` in a child Python capped at 1 GB and 30 s. Code that loops forever (as
    pps_params did before its guard) then dies in the child instead of exhausting the machine:
    on 2026-10-28 the pre-patch run of these tests reached 22 GB and the OOM killer took the
    editor with it. Returns the CompletedProcess, or None when the child timed out."""
    import resource
    import subprocess
    import sys as _s

    def _cap():
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))

    code = f"from augur import core as A\nimport math\n{expr}"
    try:
        return subprocess.run([_s.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, timeout=30, preexec_fn=_cap)
    except subprocess.TimeoutExpired:
        return None


def _refuses_with_value_error(expr: str) -> bool:
    r = _bounded_call(expr)
    return r is not None and r.returncode != 0 and "ValueError" in r.stderr


@pytest.mark.parametrize("ratio", [1.0, 1.5, 0.0, -0.5])
def test_pps_params_refuse_a_ratio_that_never_ends(ratio):
    assert _refuses_with_value_error(f"A.pps_params(ratio={ratio})")


def test_fuse_refuses_length_mismatch():
    with pytest.raises(ValueError, match="exact has 1 values for 2"):
        A.fuse(2, [], exact=A.Rung("exact", None, [0.1], 0.0))
    with pytest.raises(ValueError, match="m1 rung None has 1 values for 2"):
        A.fuse(2, [_lad("m1", [[0.4], [0.4]], [True])])


def test_calibrate_gives_a_nan_method_zero_trust():
    class _Const(A.Engine):
        def __init__(self, method, v):
            self.method, self.v = method, v

        def run(self, target, ops, param):
            return A.Rung(self.method, param, [self.v for _ in ops], 0.01)

    cal = A.calibrate([(None, [None], [0.3])],
                      [(_Const("good", 0.3), [1, 2], False), (_Const("bad", math.nan), [1, 2], False)])
    assert cal["trust"]["good"] == 1.0 and cal["trust"]["bad"] == 0.0
    with pytest.raises(ValueError, match="no method returned finite"):
        A.calibrate([(None, [None], [0.3])], [(_Const("bad", math.nan), [1, 2], False)])


@pytest.mark.parametrize("kw", [{"budget": math.nan}, {"budget": math.inf},
                                {"budget": 5, "first_rung_guess": math.nan},
                                {"budget": 5, "trust": {"pps": -1.0}},
                                {"budget": 5, "trust": {"pps": math.nan}}])
def test_campaign_refuses_budgets_and_trust_that_never_bind(kw):
    qc, _ = _ki(n=4, steps=1)
    with pytest.raises(ValueError):
        A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, _fake_engines(), log_path=None,
                   lightcone=False, **kw)
    with pytest.raises(ValueError, match="at least one engine"):
        A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [], budget=5, log_path=None,
                   lightcone=False)


def test_wrong_shaped_rung_is_still_charged():
    class _Short(A.Engine):
        method = "short"

        def run(self, target, ops, param):
            return A.Rung(self.method, param, [0.1], 0.5, degenerate=[False])

    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4), A.parse_observable("Z1", 4)]
    rep = A.Campaign({"c": (qc, ops)}, [(_Short(), [1], False)], budget=5, log_path=None,
                     lightcone=False).run()
    assert rep["spent"] == pytest.approx(0.5) and rep["steps"] == 1


def test_load_log_skips_malformed_rows_and_reports_them(tmp_path):
    import json as _j
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, _fake_engines(), budget=50,
                      log_path=None, lightcone=False)
    good = {"circuit": "c", "method": "pps", "param": 20, "values": [0.8], "elapsed_s": 1.0}
    rows = ["5", _j.dumps({**good, "values": "x"}), _j.dumps({**good, "values": ["nope"]}),
            _j.dumps({**good, "degenerate": [True, True]}), _j.dumps({**good, "elapsed_s": "NaN"}),
            _j.dumps(good)]
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(rows) + "\n")
    assert camp.load_log(p) == 1 and camp.spent == pytest.approx(1.0)
    assert "5 malformed run-log row(s)" in camp.failures[-1]["error"]


@pytest.mark.parametrize("patch", [{"budget_usd": "NaN"}, {"budget_usd": "Infinity"},
                                   {"first_rung_guess_usd": "NaN"}, {"devices": []}])
def test_day_spec_refuses_non_finite_budgets_and_empty_devices(tmp_path, patch):
    import json as _j
    f, _ = _day_spec(tmp_path)
    text = _j.dumps({**_j.loads(f.read_text()), **patch})
    f.write_text(text.replace('"NaN"', "NaN").replace('"Infinity"', "Infinity"))
    with pytest.raises(ValueError):
        A.load_day_spec(f)


def test_day_spec_must_be_an_object(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        A.load_day_spec(f)


def test_bench_refuses_an_unknown_family():
    with pytest.raises(ValueError, match="unknown bench families"):
        A.bench(families=("kicked_isng",))


def test_submission_refuses_a_report_missing_rows(tmp_path):
    spec = {"circuits": {"c": {"observables": ["Z0", "Z1"]}}}
    rep = {"circuits": {"c": {"estimates": [{"value": 0.1, "sigma": 0.01, "basis": "agree"}]}}}
    with pytest.raises(ValueError, match="1 estimates for 2"):
        A._write_submission(spec, rep, tmp_path / "s.csv")


def test_watch_unreadable_consent_spawns_nothing(tmp_path):
    af = tmp_path / "ARMED.json"
    af.write_text("{not json")
    spawned = []
    r = A.watch(armed_file=af, spawn=lambda s, o: spawned.append(s) or 1, send=_Mail())
    assert r["state"] == "armed_unreadable" and not spawned


def test_watch_survives_corrupt_state_and_shows_failed_alerts(tmp_path):
    af, out = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 4, "spent": 9, "budget": 10})
    (out / "watch_state.json").write_text("{corrupt")
    r = A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: True, spawn=lambda s, o: 1,
                send=lambda s, b: (False, "no alert channel configured"))
    assert r["state"] == "quiet"
    assert {a["kind"] for a in r["alerts"]} == {"credits80", "quiet"} and not any(a["ok"] for a in r["alerts"])


def test_pid_alive_identifies_a_process_owned_by_another_user(monkeypatch):
    import os
    import subprocess
    import sys as _s
    p = subprocess.Popen([_s.executable, "-c", "import time; time.sleep(5)", "augur", "day"])
    try:
        def _eperm(pid, sig):
            raise PermissionError(1, "Operation not permitted")
        monkeypatch.setattr(os, "kill", _eperm)
        assert A._pid_alive(p.pid) is True
    finally:
        monkeypatch.undo()
        p.kill()
        p.wait()


# ── second review round on the public code (f7572e9e7154 .. d0e1c478d925) ────────────────
def test_initialize_is_refused_every_engine_starts_from_zero():
    qc = QuantumCircuit(2)
    qc.initialize([0, 1, 0, 0], [0, 1])
    with pytest.raises(ValueError, match="initialize"):
        A.ExactEngine().run(qc, [A.parse_observable("Z0", 2)])


def test_barrier_does_not_widen_the_light_cone():
    qc = QuantumCircuit(6)
    qc.rx(0.3, 0)
    qc.barrier()
    qc.rx(0.2, 0)
    sub, _ = A._light_cone(qc, "Z", [0])
    assert sub.num_qubits == 1


def test_pps_with_no_observables_returns_an_empty_rung():
    assert A.PPSEngine().run(QuantumCircuit(2), [], 1e-3).values == []


def test_fuse_refuses_non_finite_known_answers_and_infinite_trust():
    with pytest.raises(ValueError, match="non-finite"):
        A.fuse(1, [], exact=A.Rung("exact", None, [math.nan], 0.0))
    with pytest.raises(ValueError, match="non-finite"):
        A.fuse(1, [], lightcone=A.Rung("lightcone", None, [math.inf], 0.0))
    with pytest.raises(ValueError, match="trust weights"):
        A.fuse(1, [_lad("m1", [[0.4], [0.4]], [True])], trust={"m1": math.inf})


def test_fuse_checks_every_rung_not_only_the_last():
    lad = _lad("m1", [[0.4, 0.1], [0.4]], [True])
    with pytest.raises(ValueError, match="2 values for 1"):
        A.fuse(1, [lad])


def test_run_ladder_treats_a_nan_prediction_as_unaffordable(monkeypatch):
    monkeypatch.setattr(A, "predict_next", lambda *a, **k: math.nan)
    eng = _Scripted([[0.5], [0.3], [0.31]], [0.1] * 3)
    lad = A.run_ladder(eng, None, [None], [1, 2, 3], eps_tol=1e-6)
    assert len(lad.rungs) == 1 and "predicted" in lad.stop_reason


def test_non_finite_rung_amount_is_a_failure_not_a_nan_spend():
    class _NaNCost(A.Engine):
        method = "nancost"

        def run(self, target, ops, param):
            return A.Rung(self.method, param, [0.1 for _ in ops], math.nan, degenerate=[False] * len(ops))

    qc, _ = _ki(n=4, steps=1)
    rep = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(_NaNCost(), [1, 2], False)],
                     budget=5, log_path=None, lightcone=False).run()
    assert rep["spent"] == 0.0 and "invalid amount" in rep["failures"][0]["error"]


def test_only_witness_sees_a_second_ladder_of_the_same_method():
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, _fake_engines(), budget=50,
                      log_path=None, lightcone=False)
    a, b, cloud = A.Ladder("mps"), A.Ladder("mps"), A.Ladder("pps")
    a.rungs, a.converged = [A.Rung("mps", 1, [0.2], 0.1)], [False]
    b.rungs, b.converged = [A.Rung("mps", 2, [0.3], 0.1)], [True]
    cloud.rungs = [A.Rung("pps", p, [0.0], 0.1, degenerate=[True]) for p in (1, 2)]
    camp.tracks = [A._Track("c", None, [], False, l) for l in (a, b, cloud)]
    assert camp._only_witness("c", 0, "mps")


@pytest.mark.parametrize("patch", [{"devices": ["mps.cpu", "mps.cpu"]}])
def test_day_spec_refuses_a_repeated_device(tmp_path, patch):
    import json as _j
    f, _ = _day_spec(tmp_path)
    f.write_text(_j.dumps({**_j.loads(f.read_text()), **patch}))
    with pytest.raises(ValueError, match="repeat"):
        A.load_day_spec(f)


@pytest.mark.parametrize("nq", [8.5, True, 0, "8"])
def test_day_spec_refuses_a_non_integer_qubit_count(tmp_path, nq):
    import json as _j
    f, _ = _day_spec(tmp_path)
    spec = _j.loads(f.read_text())
    spec["circuits"]["c1"]["n_qubits"] = nq
    f.write_text(_j.dumps(spec))
    with pytest.raises(ValueError, match="n_qubits"):
        A.load_day_spec(f)


def test_atomic_write_leaves_no_temp_file_on_failure(tmp_path, monkeypatch):
    import os
    target = tmp_path / "x.json"
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        A._atomic_write(target, "{}")
    assert list(tmp_path.iterdir()) == []


def test_rearming_sets_a_finished_runs_heartbeat_aside_but_never_a_live_one(tmp_path):
    import json as _j
    f, _ = _day_spec(tmp_path, deadline="2099-01-01T00:00:00+00:00")
    out, af = tmp_path / "day", tmp_path / "ARMED.json"
    out.mkdir()
    (out / "heartbeat.json").write_text(_j.dumps({"pid": 1, "done": True}))
    A.arm(f, out, armed_file=af)
    assert not (out / "heartbeat.json").exists()
    r = A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: False, spawn=lambda s, o: 4242, send=_Mail())
    assert r["state"] == "restarted" and af.exists()
    (out / "heartbeat.json").write_text(_j.dumps({"pid": 4242, "done": False, "steps": 3}))
    A.arm(f, out, armed_file=af)
    assert (out / "heartbeat.json").exists()


def test_a_restarted_run_that_stays_silent_is_reported_never_respawned(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 4, "spent": 1, "budget": 10})
    spawned, mail, t = [], _Mail(), [1e9]
    kw = dict(armed_file=af, now=lambda: t[0], spawn=lambda s, o: spawned.append(s) or 4242, send=mail)
    assert A.watch(alive=lambda p: False, **kw)["state"] == "restarted"
    assert A.watch(alive=lambda p: p == 4242, **kw)["state"] == "starting"
    t[0] += A.WATCH_STALE_S + 1
    assert A.watch(alive=lambda p: p == 4242, **kw)["state"] == "starting_quiet"
    assert len(spawned) == 1 and "[Augur] restarted run silent" in mail.sent


def test_a_failing_spawn_still_reaches_give_up(tmp_path):
    af, _ = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 4, "spent": 1, "budget": 10})

    def _boom(s, o):
        raise OSError("no python here")

    states = [A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: False, spawn=_boom,
                      send=_Mail())["state"] for _ in range(4)]
    assert states == ["spawn_failed"] * 3 + ["gave_up"]


# ── review of the full diff (a65cc3288fd4) ───────────────────────────────────────────
def test_pps_params_refuse_an_infinite_start_that_never_ends():
    assert _refuses_with_value_error("A.pps_params(delta0=math.inf)")


def test_predict_next_ignores_infinite_params():
    assert A.predict_next([math.inf, 2.0], [1.0, 2.0], 4.0, inverse=False) is None
    assert A.predict_next([1.0, 2.0], [1.0, 4.0], math.inf, inverse=False) is None


@pytest.mark.parametrize("kw", [{"eps_tol": math.nan}, {"backoff_s": math.nan}, {"agree": 2.5},
                                {"retries": 1.5}])
def test_campaign_refuses_nan_and_fractional_settings(kw):
    qc, _ = _ki(n=4, steps=1)
    with pytest.raises(ValueError):
        A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, _fake_engines(), budget=5,
                   log_path=None, lightcone=False, **kw)


def test_negative_rung_amount_never_hands_budget_back():
    class _Refund(A.Engine):
        method = "refund"

        def run(self, target, ops, param):
            return A.Rung(self.method, param, [0.1 for _ in ops], -3.0, degenerate=[False] * len(ops))

    qc, _ = _ki(n=4, steps=1)
    rep = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(_Refund(), [1, 2], False)],
                     budget=5, log_path=None, lightcone=False).run()
    assert rep["spent"] == 0.0 and "invalid amount" in rep["failures"][0]["error"]


def test_day_spec_observables_must_be_a_list(tmp_path):
    import json as _j
    f, _ = _day_spec(tmp_path)
    spec = _j.loads(f.read_text())
    spec["circuits"]["c1"]["observables"] = "Z0"
    f.write_text(_j.dumps(spec))
    with pytest.raises(ValueError, match="non-empty list"):
        A.load_day_spec(f)


def test_watch_resets_a_state_file_with_wrong_types(tmp_path):
    import json as _j
    af, out = _armed(tmp_path, hb={"pid": 1, "ts": 0, "steps": 4, "spent": 9, "budget": 10})
    (out / "watch_state.json").write_text(_j.dumps({"alerted": {"credits80": "soon"}, "sent": "x"}))
    r = A.watch(armed_file=af, now=lambda: 1e9, alive=lambda p: True, spawn=lambda s, o: 1, send=_Mail())
    assert r["state"] == "quiet"


def test_pid_alive_errs_alive_when_proc_hides_the_process(monkeypatch):
    import os
    real_read = Path.read_bytes

    def _hidden(self):
        if str(self).endswith("/cmdline"):
            raise PermissionError(13, "hidepid")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _hidden)
    assert A._pid_alive(os.getpid()) is True


# ── the hard cap: a rung is killed when the budget runs out ──────────────────
class _Slow(A.Engine):
    """Rung 1 is quick and reports 0.01 s, so the one-point guard predicts rung 2 cheap
    (0.08 s); rung 2 then sleeps `sleep_s` and reports the time it really took. Module level,
    so it pickles into the hard cap's worker."""
    method = "slow"

    def __init__(self, sleep_s):
        self.sleep_s = sleep_s

    def run(self, target, ops, param):
        import time as _t
        t = _t.perf_counter()
        if param == 2:
            _t.sleep(self.sleep_s)
        took = 0.01 if param == 1 else _t.perf_counter() - t
        return A.Rung(self.method, param, [0.1 * param for _ in ops], took,
                      degenerate=[False] * len(ops))


class _Refuses(A.Engine):
    method = "refuses"

    def run(self, target, ops, param):
        raise ValueError("the engine said no")


class _Dies(A.Engine):
    """Kills its own process mid-rung, as an out-of-memory kill would."""
    method = "dies"

    def run(self, target, ops, param):
        import os
        os._exit(3)


def _slow_campaign(sleep_s, budget, hard_cap):
    qc, _ = _ki(n=4, steps=1)
    return A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(_Slow(sleep_s), [1, 2], False)],
                      budget=budget, first_rung_guess=0.01, log_path=None, lightcone=False,
                      hard_cap=hard_cap)


def test_without_the_hard_cap_a_long_rung_overshoots():
    rep = _slow_campaign(3.0, 1.0, False).run()
    assert rep["hard_cap"] is False and rep["stopped_at_budget"] == []
    assert rep["overshoot"] > 1.5                  # the failure the cap exists to stop


def test_hard_cap_kills_a_rung_that_would_run_past_the_budget():
    import time as _t
    camp = _slow_campaign(60.0, 2.0, True)
    t = _t.monotonic()
    rep = camp.run()
    assert _t.monotonic() - t < 30                 # the 60 s rung did not run to the end
    assert rep["hard_cap"] is True
    assert rep["overshoot"] < 0.5 and rep["spent"] < 2.5
    [stop] = rep["stopped_at_budget"]
    assert stop["param"] == 2 and stop["amount"] > 1.5
    lad = rep["circuits"]["c"]["ladders"][0]
    assert lad["rungs"] == 1 and "killed at the budget" in lad["stop"]
    assert camp._worker._proc is None              # the killed child is gone, not left running


def test_hard_cap_changes_nothing_when_no_rung_runs_long():
    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable("Z0", 4), A.parse_observable("Z1", 4)]
    reps = [A.Campaign({"c": (qc, ops)}, _fake_engines(), budget=9, log_path=None,
                       lightcone=False, hard_cap=cap).run() for cap in (False, True)]
    assert reps[1]["circuits"] == reps[0]["circuits"]
    assert reps[1]["spent"] == reps[0]["spent"] and reps[1]["steps"] == reps[0]["steps"]
    assert reps[1]["stopped_at_budget"] == []


def test_hard_cap_passes_an_engine_refusal_through():
    qc, _ = _ki(n=4, steps=1)
    rep = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(_Refuses(), [1, 2], False)],
                     budget=10, log_path=None, lightcone=False, hard_cap=True).run()
    [f] = rep["failures"]
    assert "ValueError: the engine said no" in f["error"]


def test_hard_cap_with_the_real_engines_and_the_light_cone():
    qc, _ = _ki(n=6, steps=2)
    ops = [A.parse_observable(s, 6) for s in ("Z0", "Z2Z3", "X5")]
    exact = A.ExactEngine().run(qc, ops).values
    res = A.estimate(qc, ops, budget_s=60, log_path=None, hard_cap=True)
    assert res["overshoot"] == 0.0
    for row, want in zip(res["estimates"], exact):
        assert abs(row["value"] - want) < 1e-2


def test_hard_cap_refuses_what_it_cannot_cap():
    qc, _ = _ki(n=4, steps=1)
    targets = {"c": (qc, [A.parse_observable("Z0", 4)])}

    class _Local(A.Engine):                        # defined in a function: does not pickle
        method = "local"

    with pytest.raises(ValueError, match="must pickle"):
        A.Campaign(targets, [(_Local(), [1], False)], budget=5, log_path=None, hard_cap=True)
    with pytest.raises(ValueError, match="local engines only"):
        A.Campaign(targets, [(A.BlueQubitEngine(_Client([0]), "pauli-path"), [1e-3], True)],
                   budget=5, log_path=None, lightcone=False, hard_cap=True)
    with pytest.raises(ValueError, match="True or False"):
        A.Campaign(targets, A.local_engines(), budget=5, log_path=None, hard_cap="yes")


def test_hard_cap_reports_a_worker_that_dies_and_restarts_it():
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, [(_Dies(), [1, 2], False)],
                      budget=30, log_path=None, lightcone=False, hard_cap=True, retries=1,
                      backoff_s=0.0)
    rep = camp.run()
    [f] = rep["failures"]
    assert "the rung worker died (exit code 3)" in f["error"]
    assert rep["spent"] == 0.0 and rep["stopped_at_budget"] == []


def test_hard_cap_does_not_charge_the_worker_start_up():
    qc, _ = _ki(n=4, steps=1)
    camp = A.Campaign({"c": (qc, [A.parse_observable("Z0", 4)])}, A.local_engines(), budget=30,
                      log_path=None, lightcone=True, hard_cap=True)
    try:
        assert "c" in camp.lightcone            # the light cone ran in the worker
        # charged = the light cone's own time plus the pipe round trip; start-up (0.4 s warm,
        # longer cold) would push it past this
        assert camp.spent - camp.lightcone["c"].elapsed_s < 0.2
    finally:
        camp.close()


# ── the tolerance check: is eps_tol below the scale of the values it judged? ─
class _Spread(A.Engine):
    """Observable k settles at scale * (k + 1), each rung closer by half: values whose spread
    across observables is `scale`-sized."""

    def __init__(self, method, scale):
        self.method, self.scale = method, scale

    def run(self, target, ops, param):
        k = int(param) - 19
        return A.Rung(self.method, param, [self.scale * (j + 1) + self.scale * 0.1 / 2 ** k
                                           for j in range(len(ops))], 0.1,
                      degenerate=[False] * len(ops))


def _spread_report(scale, eps_tol):
    qc, _ = _ki(n=4, steps=1)
    ops = [A.parse_observable(s, 4) for s in ("Z0", "Z1", "Z2", "Z3")]
    engines = [(_Spread("pps", scale), list(range(20, 40)), False),
               (_Spread("mps", scale), list(range(20, 40)), False)]
    return A.Campaign({"c": (qc, ops)}, engines, budget=50, log_path=None, lightcone=False,
                      eps_tol=eps_tol).run()


def test_tolerance_check_flags_the_duke_case():
    # OTOCs near 1e-3 against the default 1e-2: two rungs agree within eps whatever they are
    rep = _spread_report(1e-3, 1e-2)
    tc = rep["tolerance_check"]
    assert tc["ok"] is False and "set eps_tol to" in tc["reason"]
    assert len(tc["near_zero"]) == 4                    # every observable settled below eps
    assert all(r["basis"] == "agree" for r in tc["near_zero"])


def test_tolerance_check_passes_a_tolerance_below_the_scale():
    tc = _spread_report(0.1, 1e-3)["tolerance_check"]
    assert tc["ok"] is True and tc["near_zero"] == []
    assert tc["scale"] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4]), rel=1e-2)


def test_tolerance_check_never_lists_exact_or_light_cone_values():
    circuits = {"c": {"estimates": [
        {"index": 0, "value": 0.0, "basis": "lightcone"},
        {"index": 1, "value": 1e-4, "basis": "exact"},
        {"index": 2, "value": 5e-3, "basis": "mps"},
        {"index": 3, "value": 2e-3, "basis": "unconverged:pps"},
        {"index": 4, "value": math.nan, "basis": "none"}]}}
    tc = A.tolerance_check(circuits, 1e-2)
    assert [r["index"] for r in tc["near_zero"]] == [2]     # only the ladder-settled one
    assert tc["ok"] is False


def test_tolerance_check_with_no_values_says_so():
    tc = A.tolerance_check({"c": {"estimates": [{"index": 0, "value": math.nan, "basis": "none"}]}}, 1e-2)
    assert tc["ok"] is None and "no ladder values" in tc["reason"]


def test_tolerance_check_is_not_masked_by_large_exact_values():
    # exact values spread widely; the ladder values eps_tol judged sit near 1e-3
    rows = [{"index": i, "value": v, "basis": "exact"} for i, v in enumerate([-0.9, 0.0, 0.9])]
    rows += [{"index": 3 + i, "value": v, "basis": "agree"} for i, v in enumerate([1e-3, 2e-3, 3e-3])]
    tc = A.tolerance_check({"c": {"estimates": rows}}, 1e-2)
    assert tc["judged"] == 3 and tc["ok"] is False and len(tc["near_zero"]) == 3


def test_tolerance_check_on_values_that_do_not_vary():
    rows = [{"index": i, "value": 0.5, "basis": "agree"} for i in range(3)]
    tc = A.tolerance_check({"c": {"estimates": rows}}, 1e-3)
    assert tc["ok"] is False and tc["scale"] == 0.0 and "do not vary" in tc["reason"]


@pytest.mark.parametrize("bad", [-1e-3, math.nan, math.inf, True, "0.01"])
def test_tolerance_check_refuses_a_bad_tolerance(bad):
    with pytest.raises(ValueError, match="eps_tol"):
        A.tolerance_check({}, bad)
