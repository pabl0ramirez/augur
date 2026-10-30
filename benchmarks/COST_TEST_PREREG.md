# Augur cost test, pre-registration

Written 2026-10-30, before the harness existed and before any sealed run. The README says "Whether
Augur's budget planning costs less than running everything at the top setting has not been
measured yet. We make no savings claim until a pre-registered test supports one." This is that
test. The SHA-256 of this file is pinned in `benchmarks/cost_test.py` (`PREREG_SHA256`), and the file
is anchored with OpenTimestamps (`COST_TEST_PREREG.md.ots`) before the sealed run. The sealed run
refuses to start if the digest differs or the proof does not commit to it.

## 1. Question

On circuits small enough to solve exactly, does Augur's budgeted campaign reach a fixed accuracy at
lower cost than running every circuit once at the top setting?

## 2. Circuits

Augur's own practice grid, the defaults of `augur bench`, which existed before this test.

- Kicked Ising, `kicked_ising(n, steps, theta_x, heavy_hex_edges(rows, cols))`, with steps 3, 5, 8
  and theta_x 0.3, 0.7, 1.0, nine circuits.
- Brickwork, `brickwork(n, steps, seed=steps)`, with steps 3, 5, 8, three circuits.
- Observables `local_observables(n, edges)`, the kicked Ising lattice edges for kicked Ising, the
  chain (q, q+1) for brickwork, as in `bench`.

Two sizes, each a separate tier with its own verdict.

- Tier S has rows 3 and cols 6, 18 qubits (the `bench` default).
- Tier L has rows 4 and cols 6, 24 qubits (`EXACT_MAX_QUBITS`, the largest size solved exactly).

## 3. Reference

Exact statevector values (`ExactEngine`), computed once per circuit before the first repeat and
counted in neither arm. The values and a SHA-256 per tier are recorded with the result, and
`verify --reference` computes them again.

## 4. The two arms

- The naive arm runs each circuit alone, in its own child process, once with `MPSEngine` at bond
  dimension 2048, the top of `MPS_CHIS`. No ladder, no second method, no fusion, no budget. A run
  still going 600 s after its child starts is killed. Its cost is then recorded as 600 s, which
  understates the true cost, and it does not reach the target. A run that fails before the cap
  (an error, or the memory limit below) is charged the seconds it ran, never the cap, and does not
  reach the target. Pauli path at its finest threshold is not used as a naive arm, because at 1e-4
  it did not finish one 18-qubit circuit in 120 s during development.
- The Augur arm runs all 12 circuits of a tier in one `Campaign`, in its own child process, with
  `local_engines()` (Pauli path and MPS ladders), `cost_metric="seconds"`, the default `eps_tol`
  1e-2 and `agree`, `lightcone=False`, and a budget of 300 s per circuit (3,600 s per tier), the
  default `budget_s` of `estimate`. The light cone is off because at these sizes it can return exact
  answers for observables whose cone has at most 20 qubits, which says nothing about planning, and
  `bench` leaves it off by default too. A circuit's cost is the sum of the seconds of the rungs the
  campaign ran on it (`history`, failed rungs included), plus an equal share of the campaign's own
  time, which is its wall time from `run()` start to end minus all rung seconds (planning, fusion,
  bookkeeping). If the child is killed after 7,800 s or fails in any other way, every circuit of
  that tier counts as not reached at a cost of 300 s.
- Both failure rules lean against a savings claim. A naive crash is charged only what it spent, an
  Augur crash its whole budget.

Both arms run with `OMP_NUM_THREADS=4` on the same machine, each child limited to 14 GB of address
space. The load average is recorded at the start of every arm. The exact reference for a tier must
finish within 3,600 s or the run stops before any arm runs. Only one sealed run can hold the lock
at a time.

## 5. Order and repeats

Three repeats. In each, tier S then tier L. Within a tier, the naive arm runs first in repeats 1
and 3, and Augur first in repeat 2.

## 6. The target and the rule, fixed now

A circuit reaches the target when every value is finite and the Pearson correlation with the exact
values is positive with a square of at least 0.99 (if the exact values have no spread, when every
value is within 0.01 of exact instead).

For each tier and repeat, the tier PASSES when both of these hold.
- Augur reaches the target on at least as many of the 12 circuits as the naive arm.
- Augur's median per-circuit cost is lower than the naive arm's.

The verdict for a tier is SAVES when it passes in all three repeats, otherwise NO SAVINGS CLAIM.
The tiers are not pooled. Nothing is claimed beyond these two sizes, these two families and this
machine. The rule is not relaxed, the cap, budget, target and circuits are not changed, and no
repeat is dropped or rerun after the fact.

Also reported, but not ruled on, are total seconds per arm, per-circuit costs and r^2, cuts,
Augur's labels and load averages.

## 7. What this does not show

- Costs are local seconds on one machine, not cloud credits, and depend on its load.
- The naive arm is one method at its top setting. A practitioner who picks a middle bond dimension
  by experience is a stronger baseline that this test does not include.
- Exact answers stop at 24 qubits. Augur is used on larger circuits, where nothing here applies.
- Two circuit families. Deep, strongly scrambling circuits are not in the grid.

## 8. Seen before sealing

Development probes on 2026-10-30, kicked Ising, 5 steps, theta_x 0.7, not results.
- At 18 qubits MPS at 2048 took 62.6 s and was exact. Augur's campaign with a 120 s budget spent
  91.7 s for r^2 0.99986, its Pauli-path ladder still climbing when the budget ran out.
- At 24 qubits MPS at 2048 did not finish in 900 s. Augur with a 300 s budget spent 260.9 s for
  r^2 0.99981. Exact took 47.5 s.
So the expected direction differs by tier, which is why the tiers are judged separately.

## 9. History before sealing

All on 2026-10-30, before the sealed run. Sections 1 to 8 were written before the harness. A
development run at 8 qubits then exercised every path. An AI Family code review (b05bacc92453) led to
edits that change none of the circuits, cap, budget, target, repeats or rule. They are the naive
crash rule and the Augur crash rule in section 4, Augur's own overhead added to its costs, the
memory limit, reference timeout and lock in section 4, the reference digest and its recomputation
in section 3, and failed rungs counted. The text was then reworded to the house prose rule (no
colons in prose), with no change in meaning, and pinned and stamped again.
