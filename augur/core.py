"""
augur: estimate what an ideal quantum computer would measure, without the quantum computer.

Augur estimates expectation values of Pauli observables on quantum circuits with classical
simulators, and says how far each number can be trusted. It runs each method up a ladder of
settings (bond dimension for MPS, truncation threshold for Pauli propagation), trusts a value once
successive rungs agree, fuses independent methods per observable into a value with an empirical
error bar and a label (exact, lightcone, agree, disagree, unconverged), calibrates how much to
trust each method on circuits with known answers, and spends a budget across many circuits on the
next run with the highest expected gain per cost. Local engines (exact statevector, Qiskit Aer
MPS, IBM pauli-prop, light cone) and BlueQubit cloud devices share one run() call.

It was built for the BlueQubit x DuQuantum quantum echoes hackathon (Duke University, October
2026), where about ten hidden circuits had to be estimated by circuit id and were scored by
correlation with the exact values. The machinery works on any circuit without mid-circuit
measurement; exact references stop at EXACT_MAX_QUBITS.

RELATED WORK
  * Gharibyan, Hariprakash, Mullath, Su (BlueQubit, 2025), arXiv:2507.10771: Pauli-path
    simulation with coefficient truncation run as a ladder of thresholds, one method and one
    observable at a time.
  * IBM pauli-prop (https://quantum.cloud.ibm.com/docs/en/addons/pauli-prop.md): a Pauli
    propagation engine with no convergence or budget logic.
  * Augur adds the fusion of two methods per observable with an error bar, calibration on known
    answers, and one budget allocated across many circuits by expected gain in correlation.

CONVENTIONS (measured, not assumed)
  * Pauli labels are Qiskit order, rightmost character = qubit 0. BlueQubit uses the same: a
    cpu job that flipped qubit 0 returned <ZII> = +1, <IIZ> = -1.
  * BlueQubit's pauli-path device takes ONE observable per job (the SDK's `_validate_pps_job`
    refuses more), so the cloud engine submits one async job per observable there and one job
    for all observables on the other devices.
  * Every BlueQubit query this module makes is a requested observable of the target circuit;
    nothing probes a hidden circuit's structure (pulse() and bench() run local circuits).

LIMITS (stated so nobody upgrades them)
  * Hidden circuits are run by circuit id. SDK 0.19.0b1 takes a str id in `BQClient.run` and
    gives `get_circuit_info(id)` (qubits, gate counts, depths) before any run; `day_targets`
    fills a missing n_qubits from that info.
  * With an id target there is no circuit to cut, so the light cone is skipped; the local bench
    rehearses that case with `use_lightcone=False`.
  * A converged ladder is evidence, not proof (arXiv:2507.10771 section 4: "there are no
    guarantees that finer and finer resolutions will yield the same value"). `sigma` is the
    spread between the last rungs, an empirical error bar, never a bound.
  * pauli-prop returns a truncation figure alongside the operator; on a 12-qubit prototype it was
    far larger than the true error. It is logged in the rung note and never used as the error bar.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

import os as _os
# an unset or EMPTY AUGUR_HOME both mean the default, never the current directory
DATA_DIR = Path(_os.getenv("AUGUR_HOME") or "~/.augur").expanduser()
RUN_LOG = DATA_DIR / "runs.jsonl"

#: BlueQubit's published limits (bluequbit/job_metadata_constants.py, SDK 0.18.12b1, unchanged
#: in 0.19.0b1).
BQ_MIN_PPS_THRESHOLD = 1e-5
BQ_MAX_MPS_CPU_CHI = 2048
BQ_MAX_MPS_GPU_CHI = 4000

#: The ladder the BlueQubit paper uses: delta_0 ~ 0.005, ratio 1/sqrt(2).
PPS_DELTA0 = 5e-3
PPS_RATIO = 1 / math.sqrt(2)
MPS_CHIS = (16, 32, 64, 128, 256, 512, 1024, 2048)

#: Successive agreeing rungs a ladder needs before it counts as converged. Pauli-path needs
#: three: two coarse thresholds agreeing by luck is the false convergence arXiv:2507.10771
#: section 5 describes, and the 24-qubit bench (2026-10-14) hit it on deep kicked Ising.
AGREE_BY_METHOD = {"pps": 3, "mps": 2}

#: Statevector reference is exact and cheap up to here on this box (amplitudes alone are
#: 2^24 x 16 B = 256 MiB; Qiskit's working copies add to that).
EXACT_MAX_QUBITS = 24
#: Light-cone exact is only worth it when it is fast; above this width the ladders decide.
LIGHTCONE_MAX_QUBITS = 20
#: pauli-prop preallocates max_terms x (2n bool + complex) buffers; keep one under this.
PPS_MEMORY_BYTES = 3 * 10**9


# ── observables ───────────────────────────────────────────────────────────────
_TERM_SPLIT = re.compile(r"(?<![eE])(?=[+-])")
_LEADING_NUM = re.compile(r"^(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)")


def parse_observable(spec: str, n: int):
    """'Z0', 'Z3Z4', 'X0Y1Z5', '0.5*Z0+Z1', '0.5Z0', '1e-3*X2', '2*I' -> SparsePauliOp on n
    qubits (Qiskit order). 'I' is the identity term."""
    from qiskit.quantum_info import SparsePauliOp

    terms: List[Tuple[str, List[int], float]] = []
    for piece in _TERM_SPLIT.split(spec.replace(" ", "")):
        if not piece or piece == "+":
            continue
        sign = -1.0 if piece.startswith("-") else 1.0
        raw = piece.lstrip("+-")
        coef = 1.0
        if "*" in raw:
            c, raw = raw.split("*", 1)
            coef = float(c)
        else:
            m = _LEADING_NUM.match(raw)
            if m:
                coef, raw = float(m.group(1)), raw[m.end():]
        coef *= sign
        if raw.upper() in ("I", ""):
            if raw.upper() == "I":
                terms.append(("", [], coef))
                continue
            raise ValueError(f"empty term in {spec!r}")
        paulis, idx, i = "", [], 0
        while i < len(raw):
            p = raw[i].upper()
            if p not in "XYZ":
                raise ValueError(f"bad observable term {raw!r}")
            j = i + 1
            while j < len(raw) and raw[j].isdigit():
                j += 1
            if j == i + 1:
                raise ValueError(f"missing qubit index in {raw!r}")
            q = int(raw[i + 1:j])
            if q >= n:
                raise ValueError(f"qubit {q} out of range for {n} qubits")
            if q in idx:
                raise ValueError(f"qubit {q} repeated in {raw!r}")
            paulis += p
            idx.append(q)
            i = j
        terms.append((paulis, idx, coef))
    if not terms:
        raise ValueError("empty observable")
    return SparsePauliOp.from_sparse_list(terms, n)


def to_bq_pauli_sum(op) -> List[Tuple[str, float]]:
    """SparsePauliOp -> one BlueQubit pauli_sum. Same bit order (see CONVENTIONS)."""
    out = []
    for label, c in op.to_list():
        if abs(c.imag) > 1e-12:
            raise ValueError("BlueQubit pauli_sum coefficients must be real")
        out.append((label, float(c.real)))
    return out


def circuit_key(circuit) -> str:
    """A stable short id for logging: hash of the circuit's QASM."""
    from qiskit import qasm2

    try:
        text = qasm2.dumps(circuit)
    except Exception:
        text = repr(circuit.data)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── engines: one interface, local and cloud ───────────────────────────────────
@dataclass
class Rung:
    method: str
    param: Any
    values: List[Optional[float]]   # None only from the light cone, where the cone is too wide
    elapsed_s: float
    cost_usd: float = 0.0
    note: str = ""
    #: per observable: True when this rung cannot vouch for its value (truncated to an empty
    #: operator, or the term cap bound). A degenerate rung never counts toward convergence.
    degenerate: List[bool] = field(default_factory=list)


class Engine:
    """run(target, ops, param) -> Rung. `target` is a QuantumCircuit for local engines and
    whatever BQClient.run accepts for the cloud engine."""

    method = "base"

    def run(self, target, ops: Sequence, param: Any) -> Rung:  # pragma: no cover - interface
        raise NotImplementedError


def _prepare(circuit):
    """Drop final measurements; refuse any measurement, reset or initialize that remains:
    Aer would sample a measurement silently, and every engine computes from |0...0>."""
    c = circuit.remove_final_measurements(inplace=False)
    bad = {ins.operation.name for ins in c.data} & {"measure", "reset", "initialize"}
    if bad:
        raise ValueError(f"mid-circuit {sorted(bad)} not supported: expectation from |0...0> only")
    return c


class ExactEngine(Engine):
    """Statevector. The reference, not an estimator; refuses circuits past EXACT_MAX_QUBITS."""

    method = "exact"

    def run(self, target, ops, param=None) -> Rung:
        from qiskit.quantum_info import Statevector

        if target.num_qubits > EXACT_MAX_QUBITS:
            raise ValueError(f"{target.num_qubits} qubits exceeds exact limit {EXACT_MAX_QUBITS}")
        circ = _prepare(target)
        t = time.perf_counter()
        sv = Statevector(circ)
        raw = [complex(sv.expectation_value(o)) for o in ops]
        if any(abs(v.imag) > 1e-9 for v in raw):
            raise ValueError("complex expectation value: observable is not Hermitian")
        vals: List[Optional[float]] = [v.real for v in raw]
        return Rung(self.method, None, vals, time.perf_counter() - t, degenerate=[False] * len(ops))


class MPSEngine(Engine):
    """Qiskit Aer matrix-product state at bond dimension `param`, expectation values saved in
    the simulator (no shot noise). Timing covers transpilation and simulation, because the
    budget is wall clock; building the simulator object is not timed."""

    method = "mps"

    def run(self, target, ops, param) -> Rung:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        circ = _prepare(target).copy()
        for i, o in enumerate(ops):
            circ.save_expectation_value(o, list(range(circ.num_qubits)), label=f"e{i}")
        sim = AerSimulator(method="matrix_product_state",
                           matrix_product_state_max_bond_dimension=int(param))
        t = time.perf_counter()
        tc = transpile(circ, sim, optimization_level=1, seed_transpiler=1729)
        data = sim.run(tc).result().data()
        missing = [f"e{i}" for i in range(len(ops)) if f"e{i}" not in data]
        if missing:
            raise RuntimeError(f"Aer returned no saved value for {missing}; keys {sorted(data)}")
        raw = [complex(data[f"e{i}"]) for i in range(len(ops))]
        if any(abs(v.imag) > 1e-9 for v in raw):
            raise ValueError("complex expectation value: observable is not Hermitian")
        vals: List[Optional[float]] = [v.real for v in raw]
        return Rung(self.method, int(param), vals, time.perf_counter() - t,
                    degenerate=[False] * len(ops))


class PPSEngine(Engine):
    """IBM pauli-prop, Heisenberg frame, coefficient threshold `param`. <0...0| P |0...0> is
    the coefficient sum over terms made only of I and Z. A rung is degenerate for an observable
    when truncation emptied the operator or the term cap bound (then lowering the threshold
    stops changing anything and successive rungs would agree for the wrong reason)."""

    method = "pps"
    _BASIS = ("rx", "ry", "rz", "rzz", "rxx", "ryy", "cx", "cz", "h", "s", "sdg", "x", "y", "z",
              "sx", "sxdg", "swap")

    def __init__(self, max_terms: Optional[int] = None):
        self.max_terms = max_terms

    def terms_cap(self, n: int) -> int:
        if self.max_terms:
            return int(self.max_terms)
        return int(min(2 * 10**7, PPS_MEMORY_BYTES // (2 * n + 16)))

    def run(self, target, ops, param) -> Rung:
        from qiskit import transpile
        from pauli_prop import propagate_through_circuit

        t = time.perf_counter()
        circ = transpile(_prepare(target), basis_gates=list(self._BASIS), optimization_level=1,
                         seed_transpiler=1729)
        cap = self.terms_cap(circ.num_qubits)
        vals: List[Optional[float]] = []
        degen: List[bool] = []
        sizes: List[int] = []
        worst = 0.0
        for o in ops:
            if np.abs(np.imag(o.coeffs)).max() > 1e-12:
                raise ValueError("observable has complex coefficients: not Hermitian")
            real = o.copy()
            real.coeffs = np.real(real.coeffs).astype(complex)
            po, trunc = propagate_through_circuit(real, circ, max_terms=cap, atol=float(param), frame="h")
            nz = int(np.count_nonzero(np.abs(po.coeffs) > 0))
            sizes.append(nz)
            vals.append(float(sum(c.real for lbl, c in po.label_iter() if set(lbl) <= {"I", "Z"})))
            degen.append(nz == 0 or nz >= cap)
            worst = max(worst, float(trunc))
        return Rung(self.method, float(param), vals, time.perf_counter() - t,
                    note=(f"terms kept {min(sizes, default=0)}-{max(sizes, default=0)} of cap {cap}; pauli-prop "
                          f"truncation figure {worst:.3g} (not an error bar)"),
                    degenerate=degen)


class LightConeEngine(Engine):
    """Per observable term: keep only the gates in its backward light cone, drop idle qubits,
    and evaluate exactly when the cone fits LIGHTCONE_MAX_QUBITS. Exact where it applies;
    `None` where the cone is too wide. Needs the circuit itself (not an id)."""

    method = "lightcone"

    def __init__(self, max_qubits: Optional[int] = None):
        self.max_qubits = max_qubits

    def run(self, target, ops, param=None) -> Rung:
        from qiskit.quantum_info import Statevector

        cap = self.max_qubits or LIGHTCONE_MAX_QUBITS
        base = _prepare(target)
        t = time.perf_counter()
        vals: List[Optional[float]] = []
        widths = []
        for o in ops:
            total, ok = 0.0, True
            for label, idx, coef in o.to_sparse_list():
                if not label:
                    if abs(complex(coef).imag) > 1e-12:
                        raise ValueError("complex identity coefficient: observable is not Hermitian")
                    total += float(np.real(coef))
                    continue
                sub, sub_op = _light_cone(base, label, list(idx))
                widths.append(sub.num_qubits)
                if sub.num_qubits > cap:
                    ok = False
                    break
                ev = complex(Statevector(sub).expectation_value(sub_op))
                if abs(complex(coef).imag) > 1e-12 or abs(ev.imag) > 1e-9:
                    raise ValueError("complex coefficient or expectation: observable is not Hermitian")
                total += float(np.real(coef)) * ev.real
            vals.append(total if ok else None)
        return Rung(self.method, None, vals, time.perf_counter() - t,
                    note=f"cone widths {min(widths) if widths else 0}-{max(widths) if widths else 0}",
                    degenerate=[v is None for v in vals])


class BlueQubitEngine(Engine):
    """BlueQubit cloud. `device` is one of mps.cpu, mps.gpu, pauli-path, cpu, gpu. The param is
    the bond dimension for mps.*, the truncation threshold for pauli-path, ignored for cpu/gpu.
    pauli-path: one async job per observable (the SDK allows one); other devices: one job.
    `elapsed_s` is wall time including queue; `cost_usd` is what BlueQubit charged."""

    #: Job states that mean "not now", not "no": the hackathon caps jobs in flight (5 per size
    #: class, Duke 2026-10-24) and cools GPU use down. The job is resubmitted.
    BOUNCE_STATES = frozenset({"HACKATHON_PARALLEL_JOBS_LIMIT", "HACKATHON_GPU_QUOTA_COOLDOWN"})
    #: States where the same request cannot succeed as sent: a pauli-path threshold too small for
    #: the circuit, a runtime estimate over the server's limit, or a balance the job needs and
    #: the account lacks. ValueError, so the Campaign retires the ladder instead of retrying.
    REFUSED_STATES = frozenset({"FAILED_VALIDATION", "NOT_ENOUGH_FUNDS"})
    #: Extra SDK options a day spec may add, per device. The bond dimension and threshold are
    #: never among them: those come from the ladder and are checked against BlueQubit's caps.
    EXTRA_OPTION_KEYS: ClassVar[Dict[str, frozenset]] = {"mps.cpu": frozenset(), "mps.gpu": frozenset(), "cpu": frozenset(), "gpu": frozenset(),
                         "pauli-path": frozenset({"pauli_path_circuit_transpilation_level"}),
                         "pauli-path.gpu": frozenset({"pauli_path_circuit_transpilation_level",
                                                      "pauli_path_weight_cutoff"})}

    def __init__(self, client, device: str, transpilation_level: int = 2,
                 timeout_s: Optional[float] = None, extra_options: Optional[Dict[str, Any]] = None,
                 bounce_wait_s: float = 20.0, max_bounces: int = 360,
                 sleep: Callable[[float], None] = time.sleep):
        known = {"mps.cpu": "mps", "mps.gpu": "mps", "pauli-path": "pps", "pauli-path.gpu": "pps",
                 # cpu/gpu are BlueQubit's statevector simulators; no source this module has
                 # read states they are exact, so they are never labelled `exact` (sigma 0).
                 "cpu": "statevector", "gpu": "statevector"}
        if device not in known:
            raise ValueError(f"unknown BlueQubit device {device!r}")
        if extra_options is not None and not isinstance(extra_options, dict):
            raise ValueError(f"extra_options must be a mapping, not {type(extra_options).__name__}")
        extra = dict(extra_options or {})
        bad = set(extra) - self.EXTRA_OPTION_KEYS[device]
        if bad:
            raise ValueError(f"options {sorted(bad)} not allowed on {device}; "
                             f"allowed: {sorted(self.EXTRA_OPTION_KEYS[device])}")
        lvl = extra.get("pauli_path_circuit_transpilation_level", transpilation_level)
        if isinstance(lvl, bool) or lvl not in (0, 1, 2, 3):   # the SDK's PAULI_PATH_CIRCUIT_TRANSPILATION_LEVELS
            raise ValueError(f"transpilation level must be 0, 1, 2 or 3, not {lvl!r}")
        wc = extra.get("pauli_path_weight_cutoff")
        if wc is not None and (isinstance(wc, bool) or not isinstance(wc, int) or wc < 1):
            raise ValueError(f"pauli_path_weight_cutoff must be a positive integer, not {wc!r}")
        if isinstance(bounce_wait_s, bool) \
                or not (isinstance(bounce_wait_s, (int, float)) and bounce_wait_s > 0 and math.isfinite(bounce_wait_s)) \
                or isinstance(max_bounces, bool) or not isinstance(max_bounces, int) or max_bounces < 0:
            raise ValueError("bounce_wait_s must be positive and finite, max_bounces a non-negative integer")
        if timeout_s is not None and not (isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool)
                                          and timeout_s > 0 and math.isfinite(timeout_s)):
            raise ValueError(f"timeout_s must be None or positive and finite, not {timeout_s!r}")
        self.client = client
        self.device = device
        self.extra_options = extra
        self.bounce_wait_s, self.max_bounces = bounce_wait_s, max_bounces
        self._sleep = sleep
        self.method = known[device]
        self.transpilation_level = lvl                 # the validated level, never an unchecked one
        self.timeout_s = timeout_s

    def options_for(self, param) -> Dict[str, Any]:
        # extra_options holds only EXTRA_OPTION_KEYS (checked at construction), so it can set the
        # transpilation level or the GPU weight cutoff but never the bond dimension or threshold
        if self.device.startswith("mps"):
            cap = BQ_MAX_MPS_GPU_CHI if self.device == "mps.gpu" else BQ_MAX_MPS_CPU_CHI
            if int(param) > cap:
                raise ValueError(f"bond dimension {param} exceeds {self.device} cap {cap}")
            return {"mps_bond_dimension": int(param)}
        if self.device.startswith("pauli-path"):
            if param is not None and float(param) < BQ_MIN_PPS_THRESHOLD:
                raise ValueError(f"threshold {param} below BlueQubit minimum {BQ_MIN_PPS_THRESHOLD}")
            return {"pauli_path_circuit_transpilation_level": self.transpilation_level,
                    **self.extra_options, "pauli_path_truncation_threshold": param}
        return {}

    def _submit_and_wait(self, target, ops, opts, name, deadline: float) -> List[Any]:
        """Submit, wait, and return results aligned to submission order. A bounced job (the
        hackathon cap) is cancelled and resubmitted after `bounce_wait_s`, at most `max_bounces`
        times and never past `deadline`; a refused one raises ValueError. Any other failure
        cancels what was submitted, then: a submission error is re-raised as it came, a wait
        error is raised as RuntimeError, and Ctrl-C or a kill (BaseException) is re-raised."""
        bounces = 0
        while True:
            jobs: List[Any] = []
            try:
                if self.device.startswith("pauli-path"):
                    # one job per observable: the SDK refuses more on pauli-path, and on
                    # pauli-path.gpu (SDK 0.19) the same one-at-a-time shape is the safe one
                    for o in ops:
                        jobs.append(self.client.run(target, device=self.device, pauli_sum=to_bq_pauli_sum(o),
                                                    options=opts, job_name=name, asynchronous=True))
                else:
                    jobs = _as_results(self.client.run(target, device=self.device,
                                                       pauli_sum=[to_bq_pauli_sum(o) for o in ops],
                                                       options=opts, job_name=name, asynchronous=True))
            except BaseException as e:
                # a submission failed partway, or Ctrl-C / a kill arrived: cancel what went in
                for j in jobs:
                    with contextlib.suppress(Exception):
                        self.client.cancel(j.job_id)
                if not isinstance(e, Exception):
                    raise
                if self._may_bounce(e, bounces, deadline):
                    bounces += 1
                    self._sleep(self.bounce_wait_s)
                    continue
                if self._status(e) in self.REFUSED_STATES:
                    raise ValueError(f"BlueQubit refused the request: {str(e)[:200]}") from e
                raise
            ids = [j.job_id for j in jobs]
            try:
                left = deadline - time.monotonic()
                if left <= 0:              # never hand the SDK a zero or negative timeout
                    raise TimeoutError("deadline passed before the wait began")
                done = self.client.wait(ids, timeout=left) if math.isfinite(left) else self.client.wait(ids)
                return _align(done, ids)
            except BaseException as e:
                # never leave paid jobs running: not behind a retry that would submit them again,
                # not on a mismatched reply, not on Ctrl-C or a kill (SystemExit from SIGTERM).
                # `asked` counts cancel requests the SDK accepted; it cannot confirm termination.
                asked = 0
                for j in ids:
                    with contextlib.suppress(Exception):
                        self.client.cancel(j)
                        asked += 1
                if not isinstance(e, Exception):
                    raise
                if self._may_bounce(e, bounces, deadline):
                    bounces += 1
                    self._sleep(self.bounce_wait_s)
                    continue
                if self._status(e) in self.REFUSED_STATES:
                    raise ValueError(f"BlueQubit refused the request: {str(e)[:200]}") from e
                raise RuntimeError(f"BlueQubit jobs failed or timed out, cancel requested for {asked} "
                                   f"of {len(ids)} job(s): {e}") from e

    def _may_bounce(self, e: BaseException, bounces: int, deadline: float) -> bool:
        return (self._status(e) in self.BOUNCE_STATES and bounces < self.max_bounces
                and time.monotonic() + self.bounce_wait_s < deadline)

    @staticmethod
    def _status(e: Optional[BaseException]) -> Optional[str]:
        """The first string `run_status` attribute along the exception chain (the SDK's
        BQJobNotCompleteError sets it; the message text is never read). Follows __cause__, and
        __context__ only where Python did not suppress it; each exception is visited once."""
        seen: set = set()
        while e is not None and id(e) not in seen:
            seen.add(id(e))
            st = getattr(e, "run_status", None)
            if isinstance(st, str):
                return st
            e = e.__cause__ if e.__cause__ is not None else (
                None if e.__suppress_context__ else e.__context__)
        return None

    def run(self, target, ops, param=None) -> Rung:
        if hasattr(target, "num_qubits"):
            target = _prepare(target)
        t = time.perf_counter()
        opts = self.options_for(param)
        name = f"augur_{self.device}_{param}"
        deadline = time.monotonic() + self.timeout_s if self.timeout_s else math.inf
        results = self._submit_and_wait(target, ops, opts, name, deadline)
        vals: List[Optional[float]] = []
        cost = 0.0
        for r in results:
            ev = np.asarray(r.expectation_value)
            if np.iscomplexobj(ev):
                if np.abs(ev.imag).max(initial=0.0) > 1e-9:
                    raise RuntimeError(f"BlueQubit returned a complex expectation value: {ev.ravel()[:4]}")
                ev = ev.real
            vals.extend(float(v) for v in np.asarray(ev, dtype=float).ravel())
            cost += float(getattr(r, "cost", 0.0) or 0.0)
        if len(vals) != len(ops):
            raise RuntimeError(f"BlueQubit returned {len(vals)} values for {len(ops)} observables")
        if not all(v is not None and math.isfinite(v) for v in vals):
            raise RuntimeError(f"BlueQubit returned a non-finite value: {vals}")
        id_str = ",".join(str(getattr(r, "job_id", "?")) for r in results)
        # The cloud pauli-path reports no term count, so an exact 0.0 cannot be told apart from
        # an operator truncated to nothing: it is marked degenerate and must be confirmed by MPS.
        degen = [self.method == "pps" and v == 0.0 for v in vals]
        return Rung(self.method, param, vals, time.perf_counter() - t, cost,
                    note=f"bq jobs {id_str}", degenerate=degen)


def _as_results(x) -> List[Any]:
    """BQClient.run/wait return one JobResult or a list; a dict keyed by job id is accepted too."""
    if isinstance(x, dict):
        return list(x.values())
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _align(done, ids: List[str]) -> List[Any]:
    """Results in the order the jobs were submitted (one per observable on pauli-path), so a
    reordered reply can never put a value on the wrong observable."""
    by_id = done if isinstance(done, dict) else {getattr(r, "job_id", None): r for r in _as_results(done)}
    if set(ids) - set(by_id):
        raise RuntimeError(f"BlueQubit returned results for {sorted(map(str, by_id))} not {ids}")
    return [by_id[i] for i in ids]


def _light_cone(base, label: str, idx: List[int]):
    """Backward light cone of a Pauli term, compacted to the qubits it touches.

    `label`/`idx` come from SparsePauliOp.to_sparse_list, where label[i] acts on idx[i].
    Qiskit's LightCone wraps bit_terms in a PauliGate, and PauliGate reads its label in Qiskit
    order (rightmost character on the first qarg), so the label is reversed going in. Without
    the reversal a mixed term like Z0 X1 is checked as X0 Z1 and needed gates are dropped."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import LightCone

    cone = PassManager([LightCone(bit_terms=label[::-1], indices=idx)]).run(base)
    # barriers carry no dynamics; counting their qubits would widen the cone past the cap
    gates = [ins for ins in cone.data if ins.operation.name != "barrier"]
    active = sorted({cone.find_bit(q).index for ins in gates for q in ins.qubits} | set(idx))
    remap = {q: i for i, q in enumerate(active)}
    sub = QuantumCircuit(len(active))
    for ins in gates:
        sub.append(ins.operation, [remap[cone.find_bit(q).index] for q in ins.qubits])
    sub_op = SparsePauliOp.from_sparse_list([(label, [remap[q] for q in idx], 1.0)], len(active))
    return sub, sub_op


# ── ladders and convergence ───────────────────────────────────────────────────
@dataclass
class Ladder:
    method: str
    rungs: List[Rung] = field(default_factory=list)
    converged: List[bool] = field(default_factory=list)
    stop_reason: str = ""
    done: bool = False

    def values(self) -> Optional[np.ndarray]:
        if not self.rungs:
            return None
        out: np.ndarray = np.array(self.rungs[-1].values, dtype=float)
        return out

    def sigma(self, floor: float = 1e-4) -> Optional[np.ndarray]:
        """Spread between the last two rungs, per observable, never below `floor` (two rungs
        agreeing to the last digit are not evidence of zero error); inf where either rung is
        degenerate or only one rung exists."""
        if not self.rungs:
            return None
        n = len(self.rungs[-1].values)
        if len(self.rungs) < 2:
            return np.full(n, np.inf)
        a, b = self.rungs[-1], self.rungs[-2]
        s = np.maximum(np.abs(np.array(a.values, float) - np.array(b.values, float)), floor)
        bad = np.array(_degen(a, n)) | np.array(_degen(b, n))
        s[bad] = np.inf
        out: np.ndarray = s
        return out

    def shrink(self) -> np.ndarray:
        """Per observable, how fast the spread has been falling (s_last / s_prev), clipped to
        [0, 1]; 0.5 when there is not yet a history. 1 means stuck: more rungs buy nothing."""
        n = len(self.rungs[-1].values) if self.rungs else 0
        if len(self.rungs) < 3:
            return np.full(n, 0.5)
        v = [np.array(r.values, float) for r in self.rungs[-3:]]
        s1, s0 = np.abs(v[2] - v[1]), np.abs(v[1] - v[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(s0 > 0, s1 / s0, 0.5)
        return np.clip(np.nan_to_num(r, nan=0.5), 0.0, 1.0)


def _degen(r: Rung, n: int) -> List[bool]:
    """Per-observable degenerate flags. An empty list means none flagged; any other length that
    is not n is a malformed rung and raises, so a flag can never be silently dropped."""
    if not r.degenerate:
        return [False] * n
    if len(r.degenerate) != n:
        raise ValueError(f"rung {r.method}@{r.param} has {len(r.degenerate)} degenerate flags for {n} values")
    return [bool(d) for d in r.degenerate]


def pps_params(delta0: float = PPS_DELTA0, ratio: float = PPS_RATIO,
               floor: float = BQ_MIN_PPS_THRESHOLD) -> List[float]:
    if not (0 < ratio < 1) or not (0 < delta0 < math.inf and 0 < floor < math.inf):
        raise ValueError(f"need 0 < ratio < 1 and positive delta0, floor (got {ratio}, {delta0}, {floor})")
    out, d = [], delta0
    while d >= floor:
        out.append(d)
        d *= ratio
    return out


def predict_next(params: Sequence[Any], amounts: Sequence[float], nxt: Any, inverse: bool,
                 min_amount: float = 0.02) -> Optional[float]:
    """Power-law fit log y = a + b log x (x = 1/delta for PPS, chi for MPS) on rungs large
    enough to measure; predicts the next rung's seconds or credits. arXiv:2507.10771 section 3
    reports this power law for PPS. A slope below 1 (cost not growing with resolution) is
    distrusted: the prediction is then twice the last amount. None under two usable points."""
    pts = [((1 / p if inverse else p), y) for p, y in zip(params, amounts, strict=True)
           if p is not None and 0 < p < math.inf and math.isfinite(y) and y > min_amount]
    if nxt is None or not (0 < nxt < math.inf) or len(pts) < 2 or len({x for x, _ in pts}) < 2:
        return None
    xs, ys = zip(*pts, strict=True)
    try:
        b, a = np.polyfit(np.log(xs), np.log(ys), 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if b < 1.0:
        return 2.0 * float(ys[-1])
    x = 1 / nxt if inverse else nxt
    try:
        return float(math.exp(a + b * math.log(x)))
    except OverflowError:
        return math.inf


def converged_mask(lad: Ladder, eps_tol: float, agree: int) -> List[bool]:
    """An observable is converged when the last `agree` rungs are all non-degenerate for it and
    agree within eps_tol. A true zero (an observable forbidden by symmetry) converges like any
    value; a zero produced by truncation is caught by the degenerate flag, not by its value.
    Exception: the cloud pauli-path device reports no term count, so BlueQubitEngine flags
    every exact 0.0 from it as degenerate, true zeros included; MPS must confirm those."""
    if len(lad.rungs) < agree:
        return [False] * (len(lad.rungs[-1].values) if lad.rungs else 0)
    n = len(lad.rungs[-1].values)
    window = lad.rungs[-agree:]
    vals = np.array([r.values for r in window], dtype=float)
    degen = np.array([_degen(r, n) for r in window]).any(axis=0)
    spread = vals.max(axis=0) - vals.min(axis=0)
    return [bool(s <= eps_tol and not d) for s, d in zip(spread, degen, strict=True)]


def run_ladder(engine: Engine, target, ops, params: Sequence[Any], *, eps_tol: float = 1e-2,
               agree: int = 2, budget_s: float = 300.0, inverse: bool = False,
               log: Optional[Callable[[Dict[str, Any]], None]] = None) -> Ladder:
    """One circuit, one method: climb `params` until every observable converges, the predicted
    next rung would overrun what is left of budget_s, params run out, or the engine fails.
    Finished rungs are always kept."""
    lad = Ladder(engine.method)
    spent = 0.0
    for p in params:
        if lad.rungs:
            pred = predict_next([r.param for r in lad.rungs], [r.elapsed_s for r in lad.rungs], p, inverse)
            # NaN compares false with everything: a non-finite prediction is unaffordable
            if pred is not None and (not math.isfinite(pred) or spent + pred > budget_s):
                lad.stop_reason = f"next rung {p} predicted {pred:.1f}s, {budget_s - spent:.1f}s left"
                break
        try:
            r = engine.run(target, ops, p)
        except Exception as e:
            lad.stop_reason = f"engine failed at {p}: {type(e).__name__}: {str(e)[:160]}"
            break
        lad.rungs.append(r)
        spent += r.elapsed_s
        if log:
            log(_rung_record(r))
        need_agree = max(agree, AGREE_BY_METHOD.get(engine.method, agree))
        lad.converged = converged_mask(lad, eps_tol, need_agree)
        if lad.converged and all(lad.converged):
            lad.stop_reason = f"converged within {eps_tol} over {need_agree} rungs at {p}"
            break
        if spent >= budget_s:
            lad.stop_reason = f"budget {budget_s}s spent"
            break
    if not lad.stop_reason:
        lad.stop_reason = "parameters exhausted"
    if lad.rungs and not lad.converged:
        lad.converged = [False] * len(lad.rungs[-1].values)
    lad.done = True
    return lad


def _rung_record(r: Rung) -> Dict[str, Any]:
    return {"method": r.method, "param": r.param, "elapsed_s": round(r.elapsed_s, 4),
            "cost_usd": r.cost_usd, "values": r.values, "degenerate": r.degenerate, "note": r.note}


# ── fusion ────────────────────────────────────────────────────────────────────
@dataclass
class Estimate:
    index: int
    value: float
    sigma: float
    basis: str  # exact | lightcone | agree | <method> | disagree:<m1>|<m2> | unconverged:<method>


def fuse(n_ops: int, ladders: Sequence[Ladder], lightcone: Optional[Rung] = None,
         exact: Optional[Rung] = None, disagree_k: float = 3.0,
         trust: Optional[Dict[str, float]] = None) -> List[Estimate]:
    """Per observable: exact beats light-cone-exact beats ladders. Among ladders, converged
    ones are inverse-variance averaged; if two converged methods differ by more than
    disagree_k combined sigmas, the tighter one is kept and sigma becomes the gap (a flag
    sized to the disagreement, not a statistical error bar). With no
    converged ladder the value with the smallest finite spread is kept as a Monte-Carlo-like
    estimate (arXiv:2507.10771 section 4) and flagged. A degenerate value never counts as
    converged; it is used only when it is the sole value an observable has, and is then
    labelled `degenerate:<method>` so the user sees it is a guess.

    `trust` (from `calibrate`, measured on circuits with known answers) divides each method's
    sigma before comparison, so on a family where one method has proven more accurate its
    values win a disagreement. The reported sigma of a single-method or unconverged estimate
    is that trust-divided sigma too, so a less trusted method also carries a wider error bar.
    Default: every method trusted equally.

    Every exact, light-cone and ladder rung must hold n_ops values; a mismatch raises."""
    trust = trust or {}
    bad_trust = {m: w for m, w in trust.items() if not (math.isfinite(float(w)) and float(w) >= 0)}
    if bad_trust:
        raise ValueError(f"fuse: trust weights must be finite and non-negative: {bad_trust}")
    for name, rung in (("exact", exact), ("lightcone", lightcone)):
        if rung is None:
            continue
        if len(rung.values) != n_ops:
            raise ValueError(f"fuse: {name} has {len(rung.values)} values for {n_ops} observables")
        # None means "not covered" (a cone too wide); a NaN or inf would pass as a known answer
        if any(v is not None and not math.isfinite(v) for v in rung.values):
            raise ValueError(f"fuse: {name} holds a non-finite value: {rung.values}")
    for lad in ladders:
        for r in lad.rungs:
            if len(r.values) != n_ops:
                raise ValueError(f"fuse: {lad.method} rung {r.param} has {len(r.values)} values "
                                 f"for {n_ops} observables")
    out: List[Estimate] = []
    for i in range(n_ops):
        ex_v = exact.values[i] if exact is not None else None
        if ex_v is not None:
            out.append(Estimate(i, float(ex_v), 0.0, "exact"))
            continue
        lc_v = lightcone.values[i] if lightcone is not None else None
        if lc_v is not None:
            out.append(Estimate(i, float(lc_v), 0.0, "lightcone"))
            continue
        cands: List[Tuple[str, float, float, bool, bool]] = []
        for lad in ladders:
            vals_now, sig_now = lad.values(), lad.sigma()
            if vals_now is None or sig_now is None:
                continue
            v = float(vals_now[i])
            if not math.isfinite(v):
                continue
            sig = float(sig_now[i]) / max(float(trust.get(lad.method, 1.0)), 1e-6)
            is_conv = bool(lad.converged[i]) if len(lad.converged) == n_ops else False
            last_degen = _degen(lad.rungs[-1], n_ops)[i]
            cands.append((lad.method, v, sig, is_conv and not last_degen, last_degen))
        if not cands:
            raise ValueError(f"no estimate for observable {i}")
        conv = [c for c in cands if c[3] and math.isfinite(c[2])]
        if len(conv) >= 3:
            # keep the largest group that mutually agrees; a tight outlier cannot outvote two
            # methods that agree with each other
            def _ok(p, q):
                return abs(p[1] - q[1]) <= disagree_k * math.hypot(p[2], q[2])
            groups = []
            for p in conv:
                g = [p]
                for q in sorted((q for q in conv if q is not p), key=lambda c: c[2]):
                    if all(_ok(q, m) for m in g):          # every pair agrees, not just with p
                        g.append(q)
                groups.append(g)
            best = max(groups, key=lambda g: (len(g), -min(c[2] for c in g)))
            if len(best) >= 2:
                conv = best
        if len(conv) >= 2:
            a, b = sorted(conv, key=lambda c: c[2])[:2]
            if abs(a[1] - b[1]) > disagree_k * math.hypot(a[2], b[2]):
                out.append(Estimate(i, a[1], abs(a[1] - b[1]), f"disagree:{a[0]}|{b[0]}"))
                continue
            w = np.array([1 / c[2] ** 2 for c in conv])
            v = float(np.dot(w, [c[1] for c in conv]) / w.sum())
            out.append(Estimate(i, v, float(1 / math.sqrt(w.sum())), "agree"))
        elif len(conv) == 1:
            m, v, s, _, _ = conv[0]
            out.append(Estimate(i, v, s, m))
        else:
            usable = [c for c in cands if not c[4]]
            if usable:
                m, v, s, _, _ = min(usable, key=lambda c: c[2])
                out.append(Estimate(i, v, s if math.isfinite(s) else 1.0, f"unconverged:{m}"))
            else:
                m, v, s, _, _ = min(cands, key=lambda c: c[2])
                out.append(Estimate(i, v, 1.0, f"degenerate:{m}"))
    return out


# ── scoring ───────────────────────────────────────────────────────────────────
def score(estimates: Any, exact: Any) -> Dict[str, float]:
    """Pearson correlation and signal-to-noise var(exact) / mean squared error. The organizers
    say the score is how well the estimates "correlate (signal-to-noise)"
    (the hackathon announcement) without giving the formula, so both are reported."""
    e, x = np.asarray(estimates, float), np.asarray(exact, float)
    if e.shape != x.shape:
        raise ValueError(f"score: {e.shape} estimates against {x.shape} exact values")
    mse = float(np.mean((e - x) ** 2))
    corr = float(np.corrcoef(e, x)[0, 1]) if np.std(e) > 0 and np.std(x) > 0 else float("nan")
    if not math.isfinite(mse):
        snr = float("nan")
    else:
        snr = float(np.var(x) / mse) if mse > 0 else float("inf")
    return {"pearson": corr, "snr": snr, "rmse": math.sqrt(mse), "max_abs_err": float(np.max(np.abs(e - x)))}


def calibrate(samples: Sequence[Tuple[Any, Sequence, Sequence[float]]],
              engines: Optional[Sequence[Tuple[Engine, Sequence[Any], bool]]] = None,
              budget_s: float = 60.0) -> Dict[str, Any]:
    """Measure each method on circuits whose answers are known (the day's open sample
    circuits, with exact or light-cone-exact values) and return trust weights for `fuse`.
    Each method runs its own ladder on each sample under `budget_s`; its trust is
    1 / (rmse + 1e-3), normalised so the best method has trust 1. A sample on which a method
    returned a non-finite value counts as infinite error, so that method's trust is 0."""
    engines = engines or local_engines()
    per: Dict[str, List[float]] = {}
    detail = []
    for target, ops, exact in samples:
        for eng, params, inv in engines:
            lad = run_ladder(eng, target, ops, params, budget_s=budget_s, inverse=inv,
                             agree=AGREE_BY_METHOD.get(eng.method, 2))
            if not lad.rungs:
                continue
            s = score(lad.values(), exact)
            per.setdefault(eng.method, []).append(s["rmse"] if math.isfinite(s["rmse"]) else math.inf)
            detail.append({"method": eng.method, **s, "last_param": lad.rungs[-1].param})
    if not per:
        raise ValueError("calibration produced no results")
    raw = {m: 1.0 / (float(np.mean(v)) + 1e-3) for m, v in per.items()}
    best = max(raw.values())
    if not best > 0:
        raise ValueError("calibration: no method returned finite values on every sample")
    return {"trust": {m: r / best for m, r in raw.items()}, "detail": detail}


# ── the campaign: one budget across every circuit ─────────────────────────────
@dataclass
class Action:
    circuit: str
    method: str
    param: Any
    predicted_cost: float
    gain: float
    priority: float


def plan(candidates: Sequence[Tuple[str, str, Any, float, Any, Any]],
         all_values: Sequence[float]) -> List[Action]:
    """Order candidate rungs across circuits. Each candidate is (circuit, method, param,
    predicted cost, current sigmas of that circuit's observables, expected shrink per sigma).
    Correlation is scale-free, so error is measured against S = std of all current estimates.
    Expected gain = sum over observables of (sigma_i / S)^2 * (1 - shrink_i^2): the squared
    error a rung is expected to remove. A stuck ladder (shrink 1) has zero gain.
    Priority = gain / predicted cost."""
    finite = [v for v in all_values if math.isfinite(v)]
    spread = float(np.std(finite)) if len(finite) > 1 else 1.0
    spread = spread or 1.0
    acts = []
    for circ, meth, p, cost, sigmas, shrink in candidates:
        s = np.minimum(np.asarray([x if math.isfinite(x) else spread for x in sigmas], float), spread)
        k = np.asarray(shrink, float)
        gain = float(np.sum((s / spread) ** 2 * (1 - k ** 2)))
        acts.append(Action(circ, meth, p, cost, gain, gain / max(cost, 1e-6)))
    return sorted(acts, key=lambda a: -a.priority)


@dataclass
class _Track:
    circuit: str
    engine: Engine
    params: List[Any]
    inverse: bool
    ladder: Ladder
    pos: int = 0


class _OverBudget(Exception):
    """A capped rung was killed when the budget ran out; `elapsed` is the wall time it used."""

    def __init__(self, elapsed: float):
        super().__init__(f"killed at the budget after {elapsed:.3f}s")
        self.elapsed = elapsed


def _rung_worker(conn) -> None:
    """Child side of _RungWorker: import the simulators once (not charged, as in-process
    imports never were), then run one rung per request until the pipe closes."""
    import importlib

    for mod in ("qiskit", "qiskit_aer", "pauli_prop"):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    conn.send("ready")
    while True:
        try:
            eng, target, ops, param = conn.recv()
        except EOFError:
            return
        try:
            out = (True, eng.run(target, ops, param))
        except Exception as e:          # SystemExit and KeyboardInterrupt end the child
            out = (False, e)
        try:
            conn.send(out)
        except Exception as e:          # an unpicklable result or exception: send its text
            conn.send((False, RuntimeError(f"rung result could not be returned: {type(e).__name__}: {e}")))


class _RungWorker:
    """Runs local rungs, one at a time, in one spawned child process, so a rung can be killed
    when the budget runs out. The Aer and pauli-prop calls used here take no time limit, so
    killing the process is how a rung is stopped once it starts. The child is started on
    first use (Campaign starts it before any timed call) and again after a kill or a crash;
    its start-up (importing the simulators) is not charged. It is a daemon: it ends with the
    parent if close() is never called."""

    START_TIMEOUT_S = 300.0

    def __init__(self):
        self._proc = None
        self._conn = None

    def _start(self) -> None:
        import multiprocessing as mp

        # spawn, not fork: the parent may hold OpenMP threads from Aer, unsafe across fork
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_rung_worker, args=(child,), daemon=True, name="augur-rung")
        proc.start()
        child.close()
        try:
            ok = parent.poll(self.START_TIMEOUT_S) and parent.recv() == "ready"
        except (EOFError, OSError):     # the child died before it was ready
            ok = False
        if not ok:
            proc.kill()
            proc.join()
            parent.close()
            raise RuntimeError(f"the rung worker did not start (exit code {proc.exitcode})")
        self._proc, self._conn = proc, parent

    def ensure_started(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self.close()
            self._start()

    def run(self, engine: Engine, target, ops, param, limit_s: float) -> Rung:
        """The engine's rung, or _OverBudget once `limit_s` seconds of wall time pass. An
        exception the engine raised is re-raised here (same type and message; its traceback
        stays in the child)."""
        self.ensure_started()
        t = time.perf_counter()
        self._conn.send((engine, target, list(ops), param))
        try:
            ready = self._conn.poll(max(0.0, limit_s))
            if ready:
                ok, payload = self._conn.recv()
        except (EOFError, OSError):
            code = self._proc.exitcode if self._proc is not None else None
            self.close()
            raise RuntimeError(f"the rung worker died (exit code {code})")
        if not ready:
            self.close()
            raise _OverBudget(time.perf_counter() - t)
        if ok:
            return payload
        raise payload

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.kill()
            self._proc.join()
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None


class Campaign:
    """Every (circuit, method) pair is a ladder. The campaign first takes the coarsest rung of
    each, then repeatedly runs the single rung with the highest expected gain per predicted
    cost across all circuits (a first rung whose predicted cost does not fit is skipped and
    reported), until no rung's PREDICTED cost fits what is left, nothing is
    left to gain, or the wall-clock deadline passes. One rung can overshoot its prediction,
    so `spent` can end above `budget`; the report says by how much. With `hard_cap=True`
    (a seconds budget, local engines only) every rung, the light cone included, runs in one
    reused child process that is killed when the budget runs out: the killed rung is charged the
    wall time it used and gives no values, and `overshoot` is then only the time taken to
    notice and kill it. Engines and targets must pickle for that. Budget is in seconds
    (local) or credits (`cost_metric="usd"`, cloud engines only: local rungs cost no credits,
    so a credit budget would never bind on them and is refused). A rung that raises anything
    but ValueError or MemoryError is retried up to `retries` times with backoff while the
    deadline allows; ValueError (a refused request) or MemoryError retires the ladder at once.
    Every failure is reported. A cloud engine's wait, on every device, is capped at the time
    left before the deadline, and its jobs are cancelled if the wait fails."""

    def __init__(self, targets: Dict[str, Tuple[Any, Sequence]],
                 engines: Sequence[Tuple[Engine, Sequence[Any], bool]], *, budget: float,
                 cost_metric: str = "seconds", eps_tol: float = 1e-2, agree: int = 2,
                 first_rung_guess: float = 1.0, log_path: Optional[Path] = RUN_LOG,
                 lightcone: bool = True, retries: int = 2, backoff_s: float = 5.0,
                 trust: Optional[Dict[str, float]] = None,
                 on_step: Optional[Callable[["Campaign"], None]] = None,
                 hard_cap: bool = False):
        # NaN compares false with everything, so a NaN budget would never bind: refuse it
        if not (math.isfinite(budget) and budget >= 0 and math.isfinite(first_rung_guess)
                and first_rung_guess >= 0):
            raise ValueError("budget and first_rung_guess must be finite and non-negative")
        if not engines:
            raise ValueError("a campaign needs at least one engine")
        bad_trust = {m: w for m, w in (trust or {}).items()
                     if isinstance(w, bool) or not isinstance(w, (int, float))
                     or not (math.isfinite(w) and w >= 0)}
        if bad_trust:
            raise ValueError(f"trust weights must be finite and non-negative: {bad_trust}")
        if (isinstance(agree, bool) or not isinstance(agree, int) or agree < 1
                or isinstance(retries, bool) or not isinstance(retries, int) or retries < 0
                or not (0 <= eps_tol < math.inf) or not (0 <= backoff_s < math.inf)):
            raise ValueError("need integer agree >= 1 and retries >= 0, finite non-negative eps_tol and backoff_s")
        if cost_metric not in ("seconds", "usd"):
            raise ValueError(f"cost_metric must be seconds or usd, not {cost_metric!r}")
        if cost_metric == "usd" and not all(isinstance(e, BlueQubitEngine) for e, _, _ in engines):
            raise ValueError("a credit budget needs cloud engines: local rungs cost 0 credits")
        if not isinstance(hard_cap, bool):
            raise ValueError("hard_cap must be True or False")
        if hard_cap:
            # killing a local process does not stop a cloud job from being billed
            if cost_metric != "seconds" or any(isinstance(e, BlueQubitEngine) for e, _, _ in engines):
                raise ValueError("hard_cap needs a seconds budget and local engines only")
            import pickle

            try:
                pickle.dumps(([e for e, _, _ in engines], targets, LightConeEngine()))
            except Exception as e:
                raise ValueError(f"hard_cap runs rungs in a child process, so engines and targets "
                                 f"must pickle: {type(e).__name__}: {e}") from None
        self._worker = _RungWorker() if hard_cap else None
        self.stopped: List[Dict[str, Any]] = []
        self.retries, self.backoff_s = retries, backoff_s
        self.trust = dict(trust or {})
        self._stop_at = math.inf
        self.on_step = on_step
        self.failures: List[Dict[str, Any]] = []
        self.targets = targets
        self.budget = float(budget)
        self.metric = cost_metric
        self.eps_tol, self.agree = eps_tol, agree
        self.guess = first_rung_guess
        self.spent = 0.0
        self.log_path = log_path
        self.history: List[Dict[str, Any]] = []
        self.lightcone: Dict[str, Rung] = {}
        self.tracks: List[_Track] = []
        for cid, (target, ops) in targets.items():
            if lightcone and hasattr(target, "num_qubits"):
                if self._worker is not None:
                    self._worker.ensure_started()       # start-up is never charged
                t = time.perf_counter()
                try:
                    lc = self._call(LightConeEngine(), target, ops, None)
                except _OverBudget as e:
                    self.spent += e.elapsed
                    self.stopped.append({"circuit": cid, "method": "lightcone", "param": None,
                                         "amount": e.elapsed})
                else:
                    self.lightcone[cid] = lc
                    if self.metric == "seconds":
                        self.spent += time.perf_counter() - t
                    _logger(log_path, cid)(_rung_record(lc))
            for eng, params, inv in engines:
                self.tracks.append(_Track(cid, eng, list(params), inv, Ladder(eng.method)))

    def _amount(self, r: Rung) -> float:
        return r.cost_usd if self.metric == "usd" else r.elapsed_s

    def _call(self, engine: Engine, target, ops, param) -> Rung:
        """One rung: in this process, or under the hard cap in the worker, killed when what
        is left of the budget runs out (raises _OverBudget)."""
        if self._worker is None:
            return engine.run(target, ops, param)
        left = self.budget - self.spent
        if left <= 0:
            raise _OverBudget(0.0)
        return self._worker.run(engine, target, ops, param, left)

    def close(self) -> None:
        """Stop the hard cap's worker process, if any. run() does this when it returns."""
        if self._worker is not None:
            self._worker.close()

    def _need(self, cid: str) -> List[int]:
        lc = self.lightcone.get(cid)
        return [i for i in range(len(self.targets[cid][1])) if lc is None or lc.values[i] is None]

    #: Only an exact value or two methods agreeing settles an observable. One method's
    #: convergence is not enough: on deep high-angle kicked Ising (bench 2026-10-14, 24 q)
    #: Pauli-path converged falsely at coarse thresholds, and treating that as settled kept
    #: MPS from ever checking it (campaign r 0.983 against MPS alone 0.9995).
    _SETTLED = ("exact", "lightcone", "agree")

    def _sigmas(self, cid: str) -> np.ndarray:
        """Remaining error of each observable of a circuit for planning: 0 where it is settled
        (exact, light cone, or converged), the fused sigma where it is not, inf where unknown."""
        est = self.estimates(cid, strict=False)
        out = []
        for i, e in enumerate(est):
            if e is None:
                out.append(math.inf)
            elif e.basis in self._SETTLED or self._only_witness(cid, i, e.basis):
                out.append(0.0)
            else:
                out.append(e.sigma)
        return np.array(out, float)

    def _only_witness(self, cid: str, i: int, basis: str) -> bool:
        """One method has converged on observable i and every other method that has run can
        only offer a degenerate value for it (a cloud Pauli-path zero, which reports no term
        count). No second witness can ever appear, so further rungs buy nothing: settled."""
        lads = [tr.ladder for tr in self.tracks if tr.circuit == cid and tr.ladder.rungs]
        mine = [l for l in lads if l.method == basis]
        others = [l for l in lads if l.method != basis]
        # the same method may run as more than one ladder: any of them converged is enough
        if not mine or not others or not any(len(l.converged) > i and l.converged[i] for l in mine):
            return False
        n = len(self.targets[cid][1])
        return all(all(_degen(r, n)[i] for r in l.rungs[-2:]) and len(l.rungs) >= 2 for l in others)

    def _predict(self, tr: _Track) -> Optional[float]:
        if tr.pos >= len(tr.params):
            return None
        if not tr.ladder.rungs:
            return self.guess
        p = predict_next([r.param for r in tr.ladder.rungs],
                         [self._amount(r) for r in tr.ladder.rungs], tr.params[tr.pos], tr.inverse,
                         min_amount=0.0 if self.metric == "usd" else 0.02)
        if p is not None:
            return p
        # one usable point: assume cost grows like resolution cubed (MPS is chi^3), which
        # over-predicts for PPS and keeps the first jump from overdrawing a credit budget
        last = tr.ladder.rungs[-1]
        prev_x = (1 / last.param) if tr.inverse else last.param
        next_x = (1 / tr.params[tr.pos]) if tr.inverse else tr.params[tr.pos]
        ratio = (next_x / prev_x) if prev_x else 2.0
        return max(self._amount(last), 1e-3) * max(ratio, 1.0) ** 3

    def _run(self, tr: _Track) -> None:
        cid = tr.circuit
        target, ops = self.targets[cid]
        need = self._need(cid)
        if tr.ladder.rungs:
            # once the coarsest rung exists, spend only on observables not yet settled
            # (exact, light cone, or two methods agreeing); on the cloud each one costs a job
            sig = self._sigmas(cid)
            need = [i for i in need if sig[i] > 0]
            if not need:
                tr.ladder.done, tr.ladder.stop_reason = True, "every observable settled"
                return
        if hasattr(tr.engine, "timeout_s") and math.isfinite(self._stop_at):
            left = self._stop_at - time.monotonic()
            if left < 1.0:                 # not enough time for a cloud round trip: do not start
                tr.ladder.done, tr.ladder.stop_reason = True, "deadline reached"
                return
            tr.engine.timeout_s = left
        if tr.pos >= len(tr.params):
            tr.ladder.done, tr.ladder.stop_reason = True, "parameters exhausted"
            return
        p = tr.params[tr.pos]
        tr.pos += 1
        r: Optional[Rung] = None
        err: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                r = self._call(tr.engine, target, [ops[i] for i in need], p)
                break
            except _OverBudget as e:         # killed at the hard cap: charge it, the budget is gone
                self.spent += e.elapsed
                tr.ladder.done = True
                tr.ladder.stop_reason = f"killed at the budget after {e.elapsed:.1f}s at {p}"
                rec = {"circuit": cid, "method": tr.engine.method, "param": p, "amount": e.elapsed}
                self.stopped.append(rec)
                self.history.append({**rec, "stopped": "budget"})
                return
            except (ValueError, MemoryError) as e:   # refused or out of memory: retrying cannot help
                err = e
                break
            except Exception as e:           # network, queue, timeout: worth another try
                err = e
                wait = self.backoff_s * (2 ** attempt)
                if attempt >= self.retries or time.monotonic() + wait >= self._stop_at:
                    break
                time.sleep(wait)
        if r is None:
            msg = f"engine failed at {p}: {type(err).__name__}: {str(err)[:160]}"
            tr.ladder.done, tr.ladder.stop_reason = True, msg
            self.failures.append({"circuit": cid, "method": tr.engine.method, "param": p, "error": msg})
            self.history.append({"circuit": cid, "method": tr.engine.method, "param": p, "error": msg})
            return
        if not (0 <= self._amount(r) < math.inf):
            # a NaN or inf amount would make `spent` NaN (a NaN budget check never binds), and
            # a negative one would hand budget back
            msg = f"engine reported an invalid amount {self._amount(r)!r} at {p}"
            tr.ladder.done, tr.ladder.stop_reason = True, msg
            self.failures.append({"circuit": cid, "method": tr.engine.method, "param": p, "error": msg})
            return
        if len(r.values) != len(need):
            # the rung ran and was paid for: charge it even though its values are unusable
            msg = f"engine returned {len(r.values)} values for {len(need)} observables at {p}"
            tr.ladder.done, tr.ladder.stop_reason = True, msg
            self.spent += self._amount(r)
            self.failures.append({"circuit": cid, "method": tr.engine.method, "param": p, "error": msg})
            self.history.append({"circuit": cid, "method": r.method, "param": p,
                                 "amount": self._amount(r), "error": msg})
            return
        full = _expand_rung(r, need, len(ops))
        if tr.ladder.rungs:
            # observables skipped this rung (already settled) keep this ladder's last value,
            # so fusion still sees both methods on them and they stay settled
            prev = tr.ladder.rungs[-1]
            skipped = set(self._need(cid)) - set(need)
            for i in skipped:
                full.values[i] = prev.values[i]
                full.degenerate[i] = _degen(prev, len(ops))[i]
        tr.ladder.rungs.append(full)
        self.spent += self._amount(r)
        _logger(self.log_path, cid)(_rung_record(full))
        self.history.append({"circuit": cid, "method": r.method, "param": p,
                             "amount": self._amount(r)})
        agree = max(self.agree, AGREE_BY_METHOD.get(tr.engine.method, self.agree))
        conv = converged_mask(tr.ladder, self.eps_tol, agree)
        tr.ladder.converged = conv
        if all(conv[i] for i in need):
            tr.ladder.done = True
            tr.ladder.stop_reason = f"converged at {p}"
        elif tr.pos >= len(tr.params):
            tr.ladder.done = True
            tr.ladder.stop_reason = "parameters exhausted"

    def run(self, max_steps: int = 10_000, deadline_s: Optional[float] = None) -> Dict[str, Any]:
        """deadline_s: wall-clock seconds from now after which no new rung starts (the
        submission deadline does not move for a slow queue)."""
        try:
            return self._run_all(max_steps, deadline_s)
        finally:
            self.close()

    def _run_all(self, max_steps: int, deadline_s: Optional[float]) -> Dict[str, Any]:
        stop_at = time.monotonic() + deadline_s if deadline_s is not None else math.inf
        self._stop_at = stop_at
        for tr in self.tracks:                      # bootstrap: coarsest rung everywhere
            if time.monotonic() >= stop_at:
                break
            if not tr.params:
                tr.ladder.done, tr.ladder.stop_reason = True, "no parameters"
                continue
            if tr.ladder.rungs:            # resumed from the run log: already bootstrapped
                continue
            if not self._need(tr.circuit):
                tr.ladder.done, tr.ladder.stop_reason = True, "light cone covered every observable"
                continue
            first = self._predict(tr)
            if first is not None and (not math.isfinite(first) or self.spent + first > self.budget):
                tr.ladder.done, tr.ladder.stop_reason = True, "budget before first rung"
                continue
            self._run(tr)
            self._stepped()
        for _ in range(max_steps):
            if time.monotonic() >= stop_at:
                break
            cands, owners = [], {}
            vals = [e.value for cid in self.targets for e in self.estimates(cid, strict=False)
                    if e is not None and math.isfinite(e.value)]
            for k, tr in enumerate(self.tracks):
                if tr.ladder.done or not tr.ladder.rungs:
                    continue
                cost = self._predict(tr)
                if cost is None or not math.isfinite(cost) or self.spent + cost > self.budget:
                    continue                       # NaN or inf: unaffordable, never "free"
                sig = self._sigmas(tr.circuit)
                shrink = tr.ladder.shrink() if len(tr.ladder.rungs) >= 2 else np.full(len(sig), 0.5)
                key = f"{tr.engine.method}#{k}"            # unique even if a method repeats
                cands.append((tr.circuit, key, tr.params[tr.pos], cost, sig, shrink))
                owners[(tr.circuit, key)] = tr
            acts = plan(cands, vals)
            # a rung on a method calibration distrusts buys proportionally less: without this,
            # trust only changed fusion and the budget still went to the weaker method
            # (rehearsal 2026-10-14, 24-qubit kicked Ising T=8 theta=1.0: 0.9764 -> 0.9648)
            for a in acts:
                w = float(self.trust.get(a.method.split("#")[0], 1.0))
                a.gain *= w
                a.priority *= w
            acts = sorted((a for a in acts if a.gain > 0), key=lambda a: -a.priority)
            if not acts:
                break
            self._run(owners[(acts[0].circuit, acts[0].method)])
            self._stepped()
        return self.report()

    def _stepped(self) -> None:
        """Checkpoint hook after every rung. A failing hook is reported, never fatal: losing a
        checkpoint must not stop the run that the checkpoint exists to protect."""
        if self.on_step is None:
            return
        try:
            self.on_step(self)
        except Exception as e:
            self.failures.append({"circuit": "-", "method": "on_step", "param": None,
                                  "error": f"checkpoint failed: {type(e).__name__}: {e}"})

    def load_log(self, path: Path) -> int:
        """Rebuild every ladder from a run log written by an earlier process, so a restart
        never buys a rung it already paid for. Returns the number of rungs restored; spending
        is restored too. Rows for other circuits, unknown methods, or rungs already restored
        are skipped. A malformed row (bad JSON, wrong types, wrong lengths, non-finite amounts)
        is skipped too, never fatal, and the count of such rows is reported as one failure."""
        if not path.exists():
            return 0
        restored = malformed = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cid, meth = rec.get("circuit"), rec.get("method")
                    tr = next((x for x in self.tracks
                               if x.circuit == cid and x.engine.method == meth), None)
                    if tr is None or "values" not in rec or rec.get("param") not in tr.params:
                        continue
                    k = tr.params.index(rec["param"])
                    n = len(self.targets[cid][1])
                    if k < tr.pos:
                        continue
                    if not isinstance(rec["values"], list) or len(rec["values"]) != n:
                        raise ValueError("values")
                    vals: List[Optional[float]] = [math.nan if v is None else float(v) for v in rec["values"]]
                    deg = [bool(d) for d in (rec.get("degenerate") or [False] * n)]
                    elapsed, cost = float(rec.get("elapsed_s", 0.0)), float(rec.get("cost_usd", 0.0))
                    if len(deg) != n or not (0 <= elapsed < math.inf and 0 <= cost < math.inf):
                        raise ValueError("degenerate or amounts")
                except (ValueError, TypeError, AttributeError):
                    malformed += 1
                    continue
                r = Rung(meth, rec["param"], vals, elapsed, cost, str(rec.get("note", "")), deg)
                tr.ladder.rungs.append(r)
                tr.pos = k + 1
                self.spent += self._amount(r)
                restored += 1
        for tr in self.tracks:
            if not tr.ladder.rungs:
                continue
            agree = max(self.agree, AGREE_BY_METHOD.get(tr.engine.method, self.agree))
            tr.ladder.converged = converged_mask(tr.ladder, self.eps_tol, agree)
            need = self._need(tr.circuit)
            if all(tr.ladder.converged[i] for i in need) or tr.pos >= len(tr.params):
                tr.ladder.done = True
                tr.ladder.stop_reason = "restored: converged or exhausted"
        if malformed:
            self.failures.append({"circuit": "-", "method": "load_log", "param": None,
                                  "error": f"{malformed} malformed run-log row(s) skipped in {path}"})
        return restored

    def estimates(self, cid: str, strict: bool = True) -> List[Optional[Estimate]]:
        n = len(self.targets[cid][1])
        lads = [tr.ladder for tr in self.tracks if tr.circuit == cid and tr.ladder.rungs]
        lc = self.lightcone.get(cid)
        if strict:
            return list(fuse(n, lads, lightcone=lc, trust=self.trust))
        try:
            full: List[Optional[Estimate]] = list(fuse(n, lads, lightcone=lc, trust=self.trust))
            return full
        except ValueError:
            pass
        out: List[Optional[Estimate]] = []
        for i in range(n):
            one_lc = None if lc is None else Rung(lc.method, None, [lc.values[i]], 0.0)
            one = [_slice_ladder(l, i) for l in lads]
            try:
                e = fuse(1, one, lightcone=one_lc, trust=self.trust)[0]
                e.index = i
                out.append(e)
            except ValueError:
                out.append(None)
        return out

    def report(self) -> Dict[str, Any]:
        circuits = {}
        missing = 0
        for cid in self.targets:
            est = self.estimates(cid, strict=False)
            rows = []
            for i, e in enumerate(est):
                if e is None:
                    missing += 1
                    rows.append({"index": i, "value": math.nan, "sigma": math.inf, "basis": "none"})
                else:
                    rows.append(asdict(e))
            circuits[cid] = {
                "estimates": rows,
                "ladders": [{"method": tr.engine.method, "rungs": len(tr.ladder.rungs),
                             "last_param": tr.ladder.rungs[-1].param if tr.ladder.rungs else None,
                             "stop": tr.ladder.stop_reason or ("budget or deadline" if tr.ladder.rungs
                                                                else "not run")}
                            for tr in self.tracks if tr.circuit == cid]}
        return {"budget": self.budget, "metric": self.metric, "spent": round(self.spent, 4),
                "overshoot": round(max(0.0, self.spent - self.budget), 4),
                "hard_cap": self._worker is not None, "stopped_at_budget": list(self.stopped),
                "steps": len(self.history), "missing_estimates": missing,
                "failures": self.failures, "circuits": circuits}


def _slice_ladder(lad: Ladder, i: int) -> Ladder:
    """The ladder restricted to observable i (for fusing one observable on its own)."""
    out = Ladder(lad.method, converged=[lad.converged[i]] if len(lad.converged) > i else [])
    for r in lad.rungs:
        out.rungs.append(Rung(r.method, r.param, [r.values[i]], r.elapsed_s, r.cost_usd, r.note,
                              [_degen(r, len(r.values))[i]]))
    return out


def _expand_rung(r: Rung, idx: List[int], n: int) -> Rung:
    """Re-index a rung run on a subset of observables to the full list (NaN, degenerate elsewhere)."""
    v: List[Optional[float]] = [math.nan] * n
    d = [True] * n
    rd = _degen(r, len(idx))
    for j, i in enumerate(idx):
        v[i] = r.values[j]
        d[i] = rd[j]
    return Rung(r.method, r.param, v, r.elapsed_s, r.cost_usd, r.note, d)


def local_engines() -> List[Tuple[Engine, Sequence[Any], bool]]:
    return [(PPSEngine(), pps_params(), True), (MPSEngine(), MPS_CHIS, False)]


def estimate(target, ops, *, budget_s: float = 300.0, eps_tol: float = 1e-2,
             use_lightcone: bool = True, engines=None, log_path: Optional[Path] = RUN_LOG,
             label: str = "", trust: Optional[Dict[str, float]] = None,
             hard_cap: bool = False) -> Dict[str, Any]:
    """One circuit through the campaign (light cone, then both ladders under one budget).
    hard_cap: kill a rung when the budget runs out (see Campaign)."""
    key = label or (circuit_key(target) if hasattr(target, "num_qubits") else str(target))
    camp = Campaign({key: (target, ops)}, engines or local_engines(), budget=budget_s,
                    eps_tol=eps_tol, log_path=log_path, lightcone=use_lightcone, trust=trust,
                    hard_cap=hard_cap)
    rep = camp.run()
    c = rep["circuits"][key]
    return {"circuit": key, "estimates": c["estimates"], "ladders": c["ladders"],
            "spent": rep["spent"], "overshoot": rep["overshoot"]}


def _logger(path: Optional[Path], key: str) -> Callable[[Dict[str, Any]], None]:
    def _log(rec: Dict[str, Any]) -> None:
        if path is None:
            return
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": ts, "circuit": key, **rec}, default=float) + "\n")
    return _log


# ── practice circuits (families the paper and the organizers point at) ────────
def heavy_hex_edges(rows: int, cols: int) -> List[Tuple[int, int]]:
    """A brick-wall lattice, degree <= 3 like heavy-hex: every qubit on a rows x cols grid,
    all horizontal bonds, and a vertical bond below (r, c) only where r + c is even, so the
    vertical bonds alternate columns and shift by one column on each row."""
    e = []
    for r in range(rows):
        for c in range(cols - 1):
            e.append((r * cols + c, r * cols + c + 1))
    for r in range(rows - 1):
        for c in range(cols):
            if (r + c) % 2 == 0:
                e.append((r * cols + c, (r + 1) * cols + c))
    return e


def kicked_ising(n: int, steps: int, theta_x: float, edges, theta_zz: float = -math.pi / 2):
    """IBM utility-experiment dynamics (Kim et al. 2023), as studied in arXiv:2507.10771:
    per step, RZZ(theta_zz) on every edge then RX(theta_x) on every qubit, from |0...0>."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for _ in range(steps):
        for a, b in edges:
            qc.rzz(theta_zz, a, b)
        for q in range(n):
            qc.rx(theta_x, q)
    return qc


def brickwork(n: int, layers: int, seed: int = 0):
    """Random brickwork: random RY/RZ on every qubit, CZ on alternating pairs."""
    from qiskit import QuantumCircuit

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for layer in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, math.pi)), q)
            qc.rz(float(rng.uniform(0, 2 * math.pi)), q)
        for q in range(layer % 2, n - 1, 2):
            qc.cz(q, q + 1)
    return qc


def local_observables(n: int, edges) -> List[str]:
    """Every single-qubit Z, every edge ZZ, X on every fourth qubit, and a mixed X-Z-Y term
    on the first three qubits so bit-order mistakes cannot hide behind symmetric labels."""
    mixed = ["X0Z1Y2"] if n >= 3 else []
    return ([f"Z{q}" for q in range(n)] + [f"Z{a}Z{b}" for a, b in edges]
            + [f"X{q}" for q in range(0, n, 4)] + mixed)


# ── bench: which method wins on which family, against exact ───────────────────
def bench(families: Sequence[str] = ("kicked_ising", "brickwork"), n_rows: int = 3, n_cols: int = 6,
          steps: Sequence[int] = (3, 5, 8), theta_x: Sequence[float] = (0.3, 0.7, 1.0),
          budget_s: float = 60.0, use_lightcone: bool = False,
          out: Optional[Path] = None) -> Dict[str, Any]:
    """Score the campaign and each method alone against exact on practice circuits. The
    default leaves the light cone off, because the graded circuits are hidden and the day's
    campaign cannot cut a cone it cannot see."""
    unknown = sorted(set(families) - {"kicked_ising", "brickwork"})
    if unknown:
        raise ValueError(f"unknown bench families {unknown}; known: kicked_ising, brickwork")
    n = n_rows * n_cols
    edges = heavy_hex_edges(n_rows, n_cols)
    rows = []
    for fam in families:
        grid = ([(t, th) for t in steps for th in theta_x] if fam == "kicked_ising"
                else [(t, None) for t in steps])
        for t, th in grid:
            qc = (kicked_ising(n, t, float(th), edges) if fam == "kicked_ising" and th is not None
                  else brickwork(n, t, seed=t))
            specs = local_observables(n, edges if fam == "kicked_ising" else
                                      [(q, q + 1) for q in range(n - 1)])
            ops = [parse_observable(s, n) for s in specs]
            ex = ExactEngine().run(qc, ops)
            res = estimate(qc, ops, budget_s=budget_s, use_lightcone=use_lightcone, log_path=None,
                           label=f"{fam}_T{t}_th{th}")
            fused = [e["value"] for e in res["estimates"]]
            row = {"family": fam, "n": n, "steps": t, "theta_x": th, "n_obs": len(ops),
                   "fused": score(fused, ex.values), "fused_seconds": round(res["spent"], 2),
                   "bases": _count([e["basis"] for e in res["estimates"]]),
                   "ladders": res["ladders"]}
            for name, eng, params in (("pps", PPSEngine(), pps_params()), ("mps", MPSEngine(), MPS_CHIS)):
                lad = run_ladder(eng, qc, ops, params, budget_s=budget_s, inverse=(name == "pps"))
                if not lad.rungs:
                    row[name] = {"error": lad.stop_reason}
                    continue
                row[name] = {**score(lad.values(), ex.values),
                             "last_param": lad.rungs[-1].param if lad.rungs else None,
                             "seconds": round(sum(r.elapsed_s for r in lad.rungs), 2),
                             "stop": lad.stop_reason}
            rows.append(row)
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "use_lightcone": use_lightcone,
              "budget_s": budget_s, "rows": rows}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=float))
    return report


def _count(xs):
    d: Dict[str, int] = {}
    for x in xs:
        k = x.split(":")[0]
        d[k] = d.get(k, 0) + 1
    return d


# ── the day: cloud campaign over the graded circuits ──────────────────────────
def load_day_spec(path: Path) -> Dict[str, Any]:
    """Read the day's plan:

        {"budget_usd": 25.0,
         "deadline_utc": "2026-10-25T15:20:00+00:00",      # stop starting rungs here
         "devices": ["mps.cpu", "pauli-path"],              # optional
         "circuits": {"<id>": {"target": "<id or path.qasm>", "n_qubits": 50,
                               "observables": ["Z0", "Z3Z4", "X0Y1"]}}}

    A target ending in .qasm is loaded as a circuit (the sample circuits); anything else is
    passed to BQClient.run as given (the graded ids; see LIMITS)."""
    spec: Dict[str, Any] = json.loads(Path(path).read_text())
    if not isinstance(spec, dict):
        raise ValueError("day spec must be a JSON object")
    for key in ("budget_usd", "circuits"):
        if key not in spec:
            raise ValueError(f"day spec missing {key!r}")
    # json.loads accepts NaN and Infinity, and a NaN budget would never bind
    for key in ("budget_usd", "first_rung_guess_usd"):
        v = spec.get(key, 0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{key} must be a finite non-negative number, not {v!r}")
    if "devices" in spec and not (isinstance(spec["devices"], list) and spec["devices"]):
        raise ValueError("devices, when given, must be a non-empty list")
    if "devices" in spec and len(set(map(str, spec["devices"]))) != len(spec["devices"]):
        raise ValueError(f"devices repeat {spec['devices']}: the same backend would be bought twice")
    if not isinstance(spec["circuits"], dict) or not spec["circuits"]:
        raise ValueError("day spec has no circuits")
    for cid, c in spec["circuits"].items():
        if not isinstance(c, dict):
            raise ValueError(f"circuit {cid!r} must be a JSON object")
        for key in ("target", "observables"):
            if key not in c:
                raise ValueError(f"circuit {cid!r} missing {key!r}")
        if "n_qubits" not in c and str(c["target"]).endswith(".qasm"):
            raise ValueError(f"circuit {cid!r} missing 'n_qubits' (only an id target can look it up)")
        nq = c.get("n_qubits")
        if nq is not None and (isinstance(nq, bool) or not isinstance(nq, int) or nq < 1):
            raise ValueError(f"circuit {cid!r}: n_qubits must be a positive integer, not {nq!r}")
        if not isinstance(c["observables"], list) or not c["observables"]:
            raise ValueError(f"circuit {cid!r} has no observables (a non-empty list is required)")
    if spec.get("deadline_utc"):
        from datetime import datetime

        if datetime.fromisoformat(spec["deadline_utc"]).tzinfo is None:
            raise ValueError("deadline_utc needs a timezone offset, e.g. +00:00")
    return spec


def day_targets(spec: Dict[str, Any], client: Any = None) -> Dict[str, Tuple[Any, Sequence[Any]]]:
    """Circuits and parsed observables. An id target with no n_qubits gets its size from
    BQClient.get_circuit_info (SDK 0.19+), the organizers' own pre-run metadata; the info is
    kept in spec['circuit_info'] for the report."""
    from qiskit import QuantumCircuit

    out: Dict[str, Tuple[Any, Sequence[Any]]] = {}
    for cid, c in spec["circuits"].items():
        target = c["target"]
        if "n_qubits" not in c:
            if client is None or not hasattr(client, "get_circuit_info"):
                raise ValueError(f"circuit {cid!r} has no n_qubits and no client to look it up")
            info = client.get_circuit_info(target)
            c["n_qubits"] = int(info.num_qubits)
            spec.setdefault("circuit_info", {})[cid] = {
                "num_qubits": info.num_qubits, "count_ops": dict(info.count_ops),
                "depth_1q": info.depth_1q, "depth_2q": info.depth_2q}
        if isinstance(target, str) and target.endswith(".qasm"):
            target = QuantumCircuit.from_qasm_file(target)
            if target.num_qubits != int(c["n_qubits"]):
                raise ValueError(f"{cid}: qasm has {target.num_qubits} qubits, spec says {c['n_qubits']}")
        out[cid] = (target, [parse_observable(s, int(c["n_qubits"])) for s in c["observables"]])
    return out


def run_day(spec: Dict[str, Any], client, out_dir: Path = DATA_DIR / "day",
            now: Optional[Callable[[], float]] = None, resume: bool = False) -> Dict[str, Any]:
    """Campaign on BlueQubit across every circuit of the spec, credits as the budget, the
    spec's deadline as the wall clock. Writes the full report and a submission CSV
    (circuit, observable index, observable, value, sigma, basis) and returns the report.
    Every observable gets a value: one with no estimate at the deadline is written as 0.0,
    and listed under `missing` so nobody mistakes it for a result.

    After every rung it writes `heartbeat.json` and `submission_latest.csv`, so a valid
    submission exists even if the process dies; with `resume=True` it first rebuilds the
    campaign from `runs.jsonl` and buys nothing it already bought."""
    from datetime import datetime

    devices = spec.get("devices") or ["mps.cpu", "pauli-path"]
    params: Dict[str, List[Any]] = {"mps.cpu": [c for c in MPS_CHIS if c <= BQ_MAX_MPS_CPU_CHI],
              "mps.gpu": [c for c in (*MPS_CHIS, BQ_MAX_MPS_GPU_CHI) if c <= BQ_MAX_MPS_GPU_CHI],
              "pauli-path": pps_params(), "pauli-path.gpu": pps_params()}
    # The day may set its own ladder per device (e.g. pauli-path [1e-3, 3e-4, 1e-4] when the
    # server refuses finer thresholds on these circuits, Duke 2026-10-24) and extra options per
    # device (pauli_path_circuit_transpilation_level, pauli_path_weight_cutoff).
    spec_params = spec.get("params") or {}
    dev_opts: Dict[str, Dict[str, Any]] = spec.get("device_options") or {}
    for key, val in (("params", spec_params), ("device_options", dev_opts)):
        if not isinstance(val, dict):
            raise ValueError(f"spec {key!r} must be a mapping of device to value")
        stray = set(val) - set(devices)
        if stray:
            raise ValueError(f"{key} for devices this day does not run: {sorted(stray)}")
    for d, ladder in spec_params.items():
        if d not in params:
            raise ValueError(f"params given for unsupported device {d!r}")
        if not isinstance(ladder, list) or not ladder:
            raise ValueError(f"params ladder for {d!r} must be a non-empty list")
        if d.startswith("mps"):
            # a bond dimension is a positive whole number: 64 or 64.0, never 64.5, "1e3" or True;
            # the ladder climbs (strictly ascending), since its position is the progression
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                       and x == int(x) and x >= 1 for x in ladder):
                raise ValueError(f"mps ladder for {d!r} must be positive whole numbers: {ladder}")
            if any(b <= a for a, b in itertools.pairwise(ladder)):
                raise ValueError(f"mps ladder for {d!r} must be strictly ascending: {ladder}")
            params[d] = [int(x) for x in ladder]
        else:
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                       and x > 0 for x in ladder):
                raise ValueError(f"threshold ladder for {d!r} must be positive finite numbers: {ladder}")
            if any(b >= a for a, b in itertools.pairwise(ladder)):
                raise ValueError(f"threshold ladder for {d!r} must be strictly descending: {ladder}")
            params[d] = [float(x) for x in ladder]
    if "mps.cpu" in devices and "mps.gpu" in devices:
        raise ValueError("choose mps.cpu or mps.gpu, not both: they are one method to the fusion")
    if "pauli-path" in devices and "pauli-path.gpu" in devices:
        raise ValueError("choose pauli-path or pauli-path.gpu, not both: they are one method to the fusion")
    engines: List[Tuple[Engine, Sequence[Any], bool]] = []
    for d in devices:
        if d not in params:
            raise ValueError(f"unsupported day device {d!r}")
        engines.append((BlueQubitEngine(client, d, extra_options=dev_opts.get(d)), params[d],
                        d.startswith("pauli-path")))
    deadline_s = None
    if spec.get("deadline_utc"):
        dl = datetime.fromisoformat(spec["deadline_utc"])
        deadline_s = max(0.0, dl.timestamp() - (now() if now else time.time()))
    targets = day_targets(spec, client)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint(c: "Campaign") -> None:
        _write_submission(spec, c.report(), out_dir / "submission_latest.csv")
        _write_heartbeat(out_dir, c, spec, done=False)

    # eps_tol must sit below the scale of the values: Duke 2026-10-24 G1/G2 OTOCs are ~1e-3, so
    # the default 1e-2 would call two noise-level rungs "converged" and stop the ladder there
    # (caught in review).
    raw_tol = spec.get("eps_tol", 1e-2)
    if isinstance(raw_tol, bool) or not isinstance(raw_tol, (int, float)) \
            or not (raw_tol > 0 and math.isfinite(raw_tol)):
        raise ValueError(f"eps_tol must be a positive finite number, not {raw_tol!r}")
    eps_tol = float(raw_tol)
    camp = Campaign(targets, engines, budget=float(spec["budget_usd"]), cost_metric="usd",
                    eps_tol=eps_tol,
                    first_rung_guess=float(spec.get("first_rung_guess_usd", 0.01)),
                    log_path=out_dir / "runs.jsonl", lightcone=False, trust=spec.get("trust"),
                    on_step=_checkpoint)
    restored = camp.load_log(out_dir / "runs.jsonl") if resume else 0
    _write_heartbeat(out_dir, camp, spec, done=False)
    rep = camp.run(deadline_s=deadline_s)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rep_path = out_dir / f"report_{stamp}.json"
    csv_path = out_dir / f"submission_{stamp}.csv"
    missing = _write_submission(spec, rep, csv_path)
    _write_submission(spec, rep, out_dir / "submission_latest.csv")
    rep.update({"report_path": str(rep_path), "submission_path": str(csv_path), "missing": missing,
                "restored_rungs": restored, "circuit_info": spec.get("circuit_info", {})})
    rep_path.write_text(json.dumps(rep, indent=1, default=float))
    _write_heartbeat(out_dir, camp, spec, done=True,
                     extra={"submission_path": str(csv_path), "missing": len(missing)})
    return rep


def _atomic_write(path: Path, text: str) -> None:
    import os

    import tempfile

    # a fresh, exclusively created temp file (never a predictable name that a symlink could
    # redirect), removed if anything fails before the rename
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_submission(spec: Dict[str, Any], rep: Dict[str, Any], path: Path) -> List[str]:
    """Submission CSV, written atomically. A value with no estimate is written as 0.0 and its
    id returned in the missing list."""
    import csv
    import io

    missing: List[str] = []
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["circuit", "index", "observable", "value", "sigma", "basis"])
    for cid, c in spec["circuits"].items():
        rows = rep["circuits"][cid]["estimates"]
        if len(rows) != len(c["observables"]):    # the report pads to one row per observable
            raise ValueError(f"{cid}: {len(rows)} estimates for {len(c['observables'])} observables")
        for i, e in enumerate(rows):
            v = e["value"]
            if not math.isfinite(v):
                missing.append(f"{cid}#{i}")
                v = 0.0
            w.writerow([cid, i, c["observables"][i], f"{v:.10g}", f"{e['sigma']:.3g}", e["basis"]])
    _atomic_write(path, buf.getvalue())
    return missing


def _write_heartbeat(out_dir: Path, camp: "Campaign", spec: Dict[str, Any], *, done: bool,
                     extra: Optional[Dict[str, Any]] = None) -> None:
    import os

    hb = {"pid": os.getpid(), "ts": time.time(), "spent": round(camp.spent, 6),
          "budget": camp.budget, "steps": len(camp.history), "failures": len(camp.failures),
          "deadline_utc": spec.get("deadline_utc"), "done": done, **(extra or {})}
    _atomic_write(out_dir / "heartbeat.json", json.dumps(hb))


# ── the overnight watchdog ────────────────────────────────────────────────────
#: Armed by the user only (`augur arm --confirm-spend`); the watchdog never arms itself.
ARMED_FILE = DATA_DIR / "day" / "ARMED.json"
#: A heartbeat older than this, from a run that is not done, means the run is stuck or dead.
WATCH_STALE_S = 900.0
#: Restarts allowed in a row without the run making progress (Cerebras automatic job restart:
#: progress resets the count). After this the watchdog stops and says so.
WATCH_MAX_RESTARTS_WITHOUT_PROGRESS = 3
#: No two alerts of the same kind closer than this: an alert flood hides the alert that matters.
WATCH_ALERT_GAP_S = 1800.0


def arm(spec_path: Path, out_dir: Path = DATA_DIR / "day", armed_file: Optional[Path] = None) -> Dict[str, Any]:
    """Record the user's consent for the watchdog to (re)start the paid day run. Valid only
    until the spec's deadline; refused without one."""
    from datetime import datetime

    spec = load_day_spec(spec_path)
    if not spec.get("deadline_utc"):
        raise ValueError("arming needs a deadline_utc in the spec: consent must expire")
    if datetime.fromisoformat(spec["deadline_utc"]).timestamp() <= time.time():
        raise ValueError("the spec's deadline has already passed")
    rec = {"spec": str(Path(spec_path).resolve()), "out_dir": str(Path(out_dir).resolve()),
           "deadline_utc": spec["deadline_utc"], "budget_usd": spec["budget_usd"],
           "armed_at": time.time(), "armed_by": "CLI (augur arm --confirm-spend)"}
    path = armed_file or ARMED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # a FINISHED earlier run's heartbeat would make the first watch pass report "done" and
    # disarm this new consent at once: set it and its watch state aside. A live run's
    # heartbeat stays, since hiding its pid would start a second paid run beside it.
    hb_path = Path(out_dir) / "heartbeat.json"
    try:
        stale = hb_path.exists() and bool(json.loads(hb_path.read_text()).get("done"))
    except (ValueError, AttributeError):
        stale = False
    if stale:
        stamp = int(time.time())
        for name in ("heartbeat.json", "watch_state.json"):
            old = Path(out_dir) / name
            if old.exists():
                old.replace(old.with_name(f"{name}.prev-{stamp}"))
    _atomic_write(path, json.dumps(rec, indent=1))
    return rec


def disarm(armed_file: Optional[Path] = None, reason: str = "user") -> bool:
    path = armed_file or ARMED_FILE
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
    except ValueError:
        rec = {}
    rec.update({"disarmed_at": time.time(), "disarm_reason": reason})
    _atomic_write(path.with_name("DISARMED.json"), json.dumps(rec, indent=1))
    path.unlink()
    return True


def _pid_alive(pid: Optional[int]) -> bool:
    """True only if the process exists AND is an Augur run: a recycled PID belonging to some
    other program must not make a dead run look alive (it would never be restarted).

    Errs toward alive: a live run wrongly called dead would be restarted beside itself and
    spend twice, while a dead run wrongly called alive only delays a restart. So a process
    owned by another user (EPERM) is still identified, and without /proc (macOS) an existing
    PID counts as alive. On Windows os.kill(pid, 0) would signal the process, so it refuses."""
    import os

    if os.name != "posix":
        raise RuntimeError("the watchdog's process check needs a POSIX system")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        pass                                    # it exists; someone else owns it
    except (OSError, ValueError):
        return False
    if not Path("/proc/self").exists():
        return True
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ")
    except PermissionError:
        return True                             # /proc hides it (hidepid): it exists, err alive
    except OSError:
        return False
    tokens = cmd.split()  # exact argv tokens: "augurize day" is not an Augur run
    return b"day" in tokens and any(t == b"augur" or t.endswith(b"/augur") for t in tokens)


def day_argv(spec: str, out_dir: str) -> List[str]:
    """The exact arguments the watchdog relaunches the day run with."""
    return ["day", "--spec", spec, "--out-dir", out_dir, "--confirm-spend", "--resume"]


def _spawn_day(spec: str, out_dir: str) -> int:
    """Start the day run detached from the watchdog, resuming from its own log."""
    import os
    import subprocess

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "day_process.log", "a", encoding="utf-8") as logf:
        proc = subprocess.Popen([sys.executable, "-m", "augur", *day_argv(spec, out_dir)],
                                cwd=str(Path(__file__).resolve().parents[1]), stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ))
    return proc.pid


def _email(subject: str, body: str) -> Tuple[bool, str]:
    """Send an alert by SMTP when AUGUR_SMTP_HOST, AUGUR_SMTP_USER, AUGUR_SMTP_PASSWORD and
    AUGUR_ALERT_TO are set. SSL on AUGUR_SMTP_PORT (default 465), or STARTTLS when
    AUGUR_SMTP_STARTTLS=1 (default port 587). Otherwise no alert is sent
    and the watchdog records why, so a missing channel is visible, never silent."""
    import os
    import smtplib
    import ssl
    from email.message import EmailMessage

    host, user = os.getenv("AUGUR_SMTP_HOST"), os.getenv("AUGUR_SMTP_USER")
    pwd, to = os.getenv("AUGUR_SMTP_PASSWORD"), os.getenv("AUGUR_ALERT_TO")
    if not (host and user and pwd and to):
        return False, "no alert channel configured (AUGUR_SMTP_* and AUGUR_ALERT_TO)"
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    # verify the server certificate and host name: smtplib does neither by default
    ctx = ssl.create_default_context()
    try:
        if os.getenv("AUGUR_SMTP_STARTTLS") == "1":
            with smtplib.SMTP(host, int(os.getenv("AUGUR_SMTP_PORT", "587")), timeout=30) as s:
                s.starttls(context=ctx)
                s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, int(os.getenv("AUGUR_SMTP_PORT", "465")), timeout=30, context=ctx) as s:
                s.login(user, pwd)
                s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def watch(*, armed_file: Optional[Path] = None, now: Optional[Callable[[], float]] = None,
          alive: Callable[[Optional[int]], bool] = _pid_alive,
          spawn: Callable[[str, str], int] = _spawn_day,
          send: Callable[[str, str], Tuple[bool, str]] = _email) -> Dict[str, Any]:
    """One watchdog pass. Idle unless armed. When armed: past the deadline, disarm and report;
    run finished, report once and disarm; run alive, alert on credits at 80%, on new failures,
    and when it has been quiet past WATCH_STALE_S (a live run is never restarted); run dead,
    restart it with --resume, and give up after
    WATCH_MAX_RESTARTS_WITHOUT_PROGRESS restarts that made no progress. Every alert kind is
    rate-limited to one per WATCH_ALERT_GAP_S, except the final ones."""
    from datetime import datetime

    clock = now or time.time
    path = armed_file or ARMED_FILE
    if not path.exists():
        return {"state": "idle"}
    try:
        armed = json.loads(path.read_text())
        out = Path(armed["out_dir"])
        deadline = datetime.fromisoformat(armed["deadline_utc"]).timestamp()
    except (ValueError, TypeError, KeyError, OSError) as e:
        # never guess at consent: an unreadable consent file spawns nothing and says why
        return {"state": "armed_unreadable", "error": f"{type(e).__name__}: {e}"}
    state_path = out / "watch_state.json"
    fresh = {"restarts_without_progress": 0, "steps_at_restart": -1, "alerted": {}, "failures_seen": 0}
    st: Dict[str, Any] = dict(fresh)
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text())
            ok_types = isinstance(loaded, dict) and all(
                isinstance(loaded.get(k, v), type(v)) for k, v in fresh.items()) \
                and isinstance(loaded.get("sent", []), list) \
                and all(isinstance(t, (int, float)) for t in loaded.get("alerted", {}).values())
            if ok_types:
                st.update(loaded)
            else:
                st["state_reset"] = "watch_state.json had wrong field types and was reset"
        except ValueError:
            st["state_reset"] = "watch_state.json was unreadable and was reset"
    hb: Dict[str, Any] = {}
    if (out / "heartbeat.json").exists():
        try:
            hb = json.loads((out / "heartbeat.json").read_text())
        except ValueError:
            hb = {}
    t_now = clock()
    latest = out / "submission_latest.csv"
    this_pass: List[Dict[str, Any]] = []

    def alert(kind: str, subject: str, body: str, final: bool = False) -> None:
        last = float(st["alerted"].get(kind, 0))
        if final or t_now - last >= WATCH_ALERT_GAP_S:
            ok, why = send(f"[Augur] {subject}", body)
            st["alerted"][kind] = t_now        # rate-limited even when the send failed: no flood
            rec = {"kind": kind, "ok": ok, "why": why, "ts": t_now}
            st.setdefault("sent", []).append(rec)
            this_pass.append(rec)

    def finish(state: str) -> Dict[str, Any]:
        # every alert this pass, delivered or not, is in the result, so a dead channel shows
        _atomic_write(state_path, json.dumps(st))
        return {"state": state, "heartbeat": hb, "restarts_without_progress": st["restarts_without_progress"],
                "alerts": this_pass}

    out.mkdir(parents=True, exist_ok=True)
    if hb.get("done"):
        alert("done", "run finished",
              f"Spent {hb.get('spent')} of {hb.get('budget')} credits over {hb.get('steps')} rungs. "
              f"Submission {hb.get('submission_path')}. Missing estimates {hb.get('missing')}.", final=True)
        disarm(path, "run finished")
        return finish("done")
    if t_now >= deadline:
        alert("deadline", "deadline reached",
              f"Deadline {armed['deadline_utc']} passed. Latest submission {latest} "
              f"({'exists' if latest.exists() else 'MISSING'}).", final=True)
        disarm(path, "deadline")
        return finish("deadline")
    budget = float(hb["budget"] if "budget" in hb else armed.get("budget_usd", 0))
    if budget and float(hb.get("spent", 0)) >= 0.8 * budget:
        alert("credits80", "credits at 80%", f"Spent {hb.get('spent')} of {budget}.")
    if int(hb.get("failures", 0)) > int(st.get("failures_seen", 0)):
        alert("failures", "new failures",
              f"{hb.get('failures')} failure(s) so far; see {out / 'day_process.log'}.")
        st["failures_seen"] = int(hb.get("failures", 0))
    spawned = st.get("last_spawn_pid")
    if spawned and spawned != hb.get("pid") and alive(spawned):
        # restarted, and the new run has not written its first heartbeat yet: it is starting,
        # and a second spawn now would run two paid campaigns side by side. It is never
        # restarted, but a start that stays silent past WATCH_STALE_S is reported.
        if t_now - float(st.get("last_spawn_ts", t_now)) >= WATCH_STALE_S:
            alert("start_quiet", "restarted run silent",
                  f"pid {spawned} is alive but has written no heartbeat since its restart; "
                  f"look at {out / 'day_process.log'}. Not restarted.")
            return finish("starting_quiet")
        return finish("starting")
    if alive(hb.get("pid")):
        # A live process is never restarted: one cloud rung can legitimately outlast any
        # heartbeat interval, and a second run beside it would spend credits twice.
        if int(hb.get("steps", 0)) > int(st["steps_at_restart"]):
            st["restarts_without_progress"] = 0
        if t_now - float(hb.get("ts", 0)) >= WATCH_STALE_S:
            alert("quiet", "run quiet",
                  f"No heartbeat for {int((t_now - float(hb.get('ts', 0))) / 60)} min; the process "
                  f"(pid {hb.get('pid')}) is alive, probably inside a long cloud rung. Not restarted.")
            return finish("quiet")
        return finish("running")
    steps = int(hb.get("steps", -1))
    if steps > int(st["steps_at_restart"]):
        st["restarts_without_progress"] = 0
    if st["restarts_without_progress"] >= WATCH_MAX_RESTARTS_WITHOUT_PROGRESS:
        alert("gave_up", "watchdog gave up",
              f"{st['restarts_without_progress']} restarts in a row made no progress. Run stopped; "
              f"latest submission {latest}. Look at {out / 'day_process.log'}.", final=True)
        disarm(path, "no progress after restarts")
        return finish("gave_up")
    st["restarts_without_progress"] += 1          # counted before the spawn, so a spawn that
    st["steps_at_restart"] = steps                # keeps failing still reaches the give-up
    try:
        pid = spawn(armed["spec"], armed["out_dir"])
    except Exception as e:
        alert("spawn_failed", "restart failed", f"Could not restart the day run: {type(e).__name__}: {e}")
        return finish("spawn_failed")
    st["last_spawn_pid"] = pid
    st["last_spawn_ts"] = t_now
    alert("restart", "run restarted",
          f"The day run was not running; restarted with --resume (pid {pid}). "
          f"Restart {st['restarts_without_progress']} without progress.")
    return finish("restarted")


def watch_pulse() -> Dict[str, Any]:
    """One watchdog pass as a health record. Idle (no spend, no network) unless armed."""
    try:
        r = watch()
        return {"agent": "AugurWatch", "success": r["state"] not in ("gave_up", "armed_unreadable"),
                "message": f"AugurWatch: {r['state']}"}
    except Exception as e:  # pragma: no cover
        return {"agent": "AugurWatch", "success": False, "message": f"{type(e).__name__}: {e}"}


# ── status and self-check ─────────────────────────────────────────────────────────────
def status() -> Dict[str, Any]:
    n = 0
    if RUN_LOG.exists():
        with RUN_LOG.open(encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
    return {"product": "Augur observable estimator", "run_log_rows": n,
            "honesty": "sigma is the spread between ladder rungs, an empirical error bar, never a bound"}


def pulse() -> Dict[str, Any]:
    """On a 10-qubit kicked Ising (small enough to be exact) with a mixed X-Z-Y observable:
    MPS and PPS each within 1e-2 of exact, the light cone equal to exact, a chi=2 MPS visibly
    wrong (the bench can see a bad method), and the campaign with no light cone correlating
    above 0.999."""
    try:
        n = 10
        edges = [(i, i + 1) for i in range(n - 1)]
        qc = kicked_ising(n, 5, 0.9, edges)
        ops = [parse_observable(s, n) for s in local_observables(n, edges)]
        ex = np.array(ExactEngine().run(qc, ops).values)
        mps = np.array(MPSEngine().run(qc, ops, 32).values)
        pps = np.array(PPSEngine().run(qc, ops, 1e-5).values)
        lc = np.array(LightConeEngine().run(qc, ops).values, dtype=float)
        bad = np.array(MPSEngine().run(qc, ops, 2).values)
        res = estimate(qc, ops, budget_s=60, use_lightcone=False, log_path=None, label="pulse")
        fused = score([e["value"] for e in res["estimates"]], ex)["pearson"]
        e_m, e_p, e_l, e_b = (np.abs(x - ex).max() for x in (mps, pps, lc, bad))
        ok = e_m < 1e-2 and e_p < 1e-2 and e_l < 1e-9 and e_b > 1e-3 and fused > 0.999
        return {"agent": "Augur", "success": bool(ok),
                "message": (f"mps err {e_m:.1e}, pps err {e_p:.1e}, lightcone err {e_l:.1e}, "
                            f"chi=2 err {e_b:.1e} (must be visible), campaign pearson {fused:.5f}")}
    except Exception as e:  # pragma: no cover
        return {"agent": "Augur", "success": False, "message": f"{type(e).__name__}: {e}"}


def _load_env() -> None:
    """Read ./.env if python-dotenv is installed; environment variables work without it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env")  # exactly ./.env, never a parent directory


def _cli(argv: Optional[List[str]] = None) -> int:
    ap = _parser()
    return _dispatch(ap, ap.parse_args(argv))


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="augur", description="Estimate observables without the quantum computer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="score each method against exact on practice circuits")
    b.add_argument("--rows", type=int, default=3)
    b.add_argument("--cols", type=int, default=6)
    b.add_argument("--budget-s", type=float, default=60.0)
    b.add_argument("--lightcone", action="store_true", help="allow the light cone (visible circuits only)")
    b.add_argument("--out", type=Path, default=DATA_DIR / "bench_latest.json")
    e = sub.add_parser("estimate", help="estimate observables of a QASM circuit locally")
    e.add_argument("--qasm", type=Path, required=True)
    e.add_argument("--obs", required=True, help="comma-separated, e.g. Z0,Z1Z2,X5")
    e.add_argument("--budget-s", type=float, default=300.0)
    e.add_argument("--hard-cap", action="store_true",
                   help="kill a rung when the budget runs out, so spending never passes it")
    dy = sub.add_parser("day", help="run the cloud campaign on the day's circuits (spends credits)")
    dy.add_argument("--spec", type=Path, required=True)
    dy.add_argument("--confirm-spend", action="store_true",
                    help="required: this command spends BlueQubit credits")
    dy.add_argument("--out-dir", type=Path, default=DATA_DIR / "day")
    dy.add_argument("--resume", action="store_true", help="rebuild from runs.jsonl; buy nothing twice")
    ar = sub.add_parser("arm", help="let the watchdog (re)start the paid day run until the deadline")
    ar.add_argument("--spec", type=Path, required=True)
    ar.add_argument("--out-dir", type=Path, default=DATA_DIR / "day")
    ar.add_argument("--confirm-spend", action="store_true")
    sub.add_parser("disarm", help="stop the watchdog from (re)starting the day run")
    wa = sub.add_parser("watch", help="one watchdog pass, or --loop to keep watching")
    wa.add_argument("--loop", action="store_true")
    wa.add_argument("--every-s", type=float, default=120.0)
    sub.add_parser("status")
    sub.add_parser("pulse")
    return ap


def _dispatch(ap: argparse.ArgumentParser, a: argparse.Namespace) -> int:
    if a.cmd == "arm":
        if not a.confirm_spend:
            print("refused: arming lets the watchdog spend BlueQubit credits; add --confirm-spend")
            return 2
        print(json.dumps(arm(a.spec, a.out_dir), indent=1))
        return 0
    if a.cmd == "disarm":
        print("disarmed" if disarm() else "was not armed")
        return 0
    if a.cmd == "watch":
        _load_env()
        while True:
            try:
                r = watch()
            except Exception as e:      # a looping watchdog reports a bad pass and keeps watching
                r = {"state": "error", "error": f"{type(e).__name__}: {e}"}
            print(json.dumps({k: r[k] for k in r if k != "heartbeat"}, default=str), flush=True)
            if not a.loop:
                return 1 if r["state"] in ("error", "armed_unreadable") else 0
            if r["state"] in ("idle", "done", "deadline", "gave_up"):
                return 0
            time.sleep(a.every_s)
    if a.cmd == "day":
        if not a.confirm_spend:
            print("refused: `day` spends BlueQubit credits; rerun with --confirm-spend")
            return 2
        import os

        import bluequbit

        _load_env()
        key = os.getenv("BLUEQUBIT_API_TOKEN")
        if not key:
            print("refused: no BlueQubit key (set BLUEQUBIT_API_TOKEN, or put it in ./.env)")
            return 2
        import signal

        def _term(signum, frame):
            raise SystemExit(143)          # unwinds through the engine, which cancels its jobs

        signal.signal(signal.SIGTERM, _term)
        rep = run_day(load_day_spec(a.spec), bluequbit.init(key), out_dir=a.out_dir, resume=a.resume)
        print(json.dumps({k: rep[k] for k in ("spent", "overshoot", "steps", "missing_estimates",
                                              "failures", "report_path", "submission_path", "missing",
                                              "restored_rungs")},
                         indent=1, default=float))
        return 0
    if a.cmd == "bench":
        rep = bench(n_rows=a.rows, n_cols=a.cols, budget_s=a.budget_s, use_lightcone=a.lightcone, out=a.out)
        def _m(x):
            return f"r={x['pearson']:.4f} ({x['seconds']}s)" if "pearson" in x else f"error {x['error']}"

        for r in rep["rows"]:
            print(f"{r['family']:13s} T={r['steps']:<2} th={r['theta_x']!s:5s} "
                  f"campaign r={r['fused']['pearson']:.4f} ({r['fused_seconds']}s) | "
                  f"pps {_m(r['pps'])} | mps {_m(r['mps'])} | bases {r['bases']}")
        print(f"report: {a.out}")
    elif a.cmd == "estimate":
        from qiskit import QuantumCircuit

        qc = QuantumCircuit.from_qasm_file(str(a.qasm))
        ops = [parse_observable(s, qc.num_qubits) for s in a.obs.split(",")]
        print(json.dumps(estimate(qc, ops, budget_s=a.budget_s, hard_cap=a.hard_cap), indent=1,
                         default=float))
    elif a.cmd == "status":
        print(json.dumps(status(), indent=1))
    elif a.cmd == "pulse":
        r = pulse()
        print(json.dumps(r, indent=1))
        return 0 if r["success"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
