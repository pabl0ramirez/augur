# Augur cost test, result

Run 2026-10-30 from 08:58 to 18:13 (UTC-6) under the pre-registration in
[COST_TEST_PREREG.md](COST_TEST_PREREG.md), SHA-256
`70612a82d0c7b5d393f494cb67521ba1d94f7aa528dff95e393868a72206231d`, anchored with
OpenTimestamps ([COST_TEST_PREREG.md.ots](COST_TEST_PREREG.md.ots)) before the run. The harness
refuses a sealed run unless that file has this digest and the proof commits to it. The sealed
result is [results/RESULT_70612a82d0c7.json](results/RESULT_70612a82d0c7.json), SHA-256
`3e03d4fbfaa78140cbaa985e2c07547269199329a92ed58acd78711a7708e5e9`. It holds every per-circuit
value, cost, ladder and label. From the repository root,

    PYTHONPATH=. python benchmarks/cost_test.py verify benchmarks/results/RESULT_70612a82d0c7.json --reference

recomputes the exact reference values, checks them against the digests recorded per tier,
re-judges every circuit and repeat from the recorded values, and re-derives the verdicts. On the
machine that ran the test it printed `"ok": true` with an empty list of problems and the verdicts
below.

## Verdict

**No savings claim at either size.** A tier passes a repeat only when Augur reaches the target
on at least as many circuits as the naive arm and its median cost is lower. The verdict needed a
pass in all three repeats. Augur passed in none at 18 qubits and in two of three at 24 qubits.

| Tier | Repeat | Reached, naive | Reached, Augur | Median s, naive | Median s, Augur | Passes |
|---|---|---|---|---|---|---|
| S, 18 qubits | 1 | 12 | 12 | 6.26 | 15.87 | no |
| S, 18 qubits | 2 | 12 | 12 | 6.28 | 13.98 | no |
| S, 18 qubits | 3 | 12 | 12 | 6.23 | 10.37 | no |
| L, 24 qubits | 1 | 7 | 12 | 34.15 | 50.88 | no |
| L, 24 qubits | 2 | 7 | 12 | 34.15 | 32.83 | yes |
| L, 24 qubits | 3 | 7 | 12 | 34.12 | 27.60 | yes |

## Per circuit

Median charged seconds over the three repeats, and the lowest r^2 against the exact values over
the three repeats ("cut" means the naive run was killed at 600 s in every repeat and charged
600 s, which understates what it would have cost to finish).

| Circuit | 18 q naive s | 18 q naive r^2 | 18 q Augur s | 18 q Augur r^2 | 24 q naive s | 24 q naive r^2 | 24 q Augur s | 24 q Augur r^2 |
|---|---|---|---|---|---|---|---|---|
| bw_T3 | 0.8 | 1.00000 | 8.3 | 0.99999 | 0.8 | 1.00000 | 18.3 | 0.99999 |
| bw_T5 | 0.8 | 1.00000 | 5.1 | 1.00000 | 0.8 | 1.00000 | 27.9 | 0.99999 |
| bw_T8 | 0.8 | 1.00000 | 8.6 | 1.00000 | 0.8 | 1.00000 | 15.8 | 1.00000 |
| ki_T3_th0.3 | 1.0 | 1.00000 | 4.1 | 0.99999 | 1.4 | 1.00000 | 18.8 | 0.99999 |
| ki_T3_th0.7 | 1.4 | 1.00000 | 12.2 | 1.00000 | 2.0 | 1.00000 | 24.4 | 1.00000 |
| ki_T3_th1.0 | 1.4 | 1.00000 | 15.7 | 0.99999 | 2.1 | 1.00000 | 29.7 | 0.99994 |
| ki_T5_th0.3 | 11.1 | 1.00000 | 5.5 | 0.99995 | 66.2 | 1.00000 | 36.3 | 0.99994 |
| ki_T5_th0.7 | 60.9 | 1.00000 | 89.4 | 0.99985 | 600.0 | cut | 680.7 | 0.99991 |
| ki_T5_th1.0 | 60.1 | 1.00000 | 850.3 | 0.99948 | 600.0 | cut | 645.4 | 0.99560 |
| ki_T8_th0.3 | 50.2 | 1.00000 | 25.1 | 0.99994 | 600.0 | cut | 95.4 | 0.99994 |
| ki_T8_th0.7 | 116.9 | 1.00000 | 684.0 | 0.99833 | 600.0 | cut | 1248.3 | 0.99733 |
| ki_T8_th1.0 | 121.3 | 1.00000 | 1834.2 | 0.99360 | 600.0 | cut | 756.2 | 0.99188 |

## What else the run shows (reported, not ruled on)

- At 18 qubits the top MPS setting reached the target on every circuit for about 428 charged
  seconds per repeat in all. Augur's campaign spent 3,596 to 4,204 s per repeat, its whole budget
  or more, most of it on the hardest kicked Ising circuits.
- At 24 qubits the naive arm was cut on five kicked Ising circuits in every repeat, for about
  3,074 charged seconds in all. The true cost of finishing those five is unknown and higher.
  Augur reached the target on all 12 circuits in every repeat, for about 3,590 s in all.
- Augur's median cost moved between repeats (50.9, 32.8 and 27.6 s at 24 qubits) while the naive
  arm's did not. The harness keeps nothing between repeats. Each arm runs in a fresh process, with
  no run log (`log_path=None`), no trust weights, the light cone off and nothing calibrated from
  the exact values, which the campaign never sees. The likeliest source is a plan that reacts to
  measured seconds, so small timing differences change which rungs it buys. This run does not
  isolate that cause.
- The budget is not a hard cap. In repeat 3 at 18 qubits the campaign spent 4,204 s against
  3,600 s, 17% over. The `Campaign` docstring says one rung can overshoot its prediction, and a
  local rung is not stopped once it starts. The overrun was charged to Augur in full, so it can
  only weaken a savings claim, never create one. Bounding it is future work.
- Load average at the start of each arm was between 1.0 and 1.4 on 8 CPUs, except 4.6 at the
  start of the first naive arm, which followed the exact reference computation. The order was
  naive first in repeats 1 and 3 and Augur first in repeat 2, as sealed. Python 3.12.3, numpy
  2.5.3, Qiskit 2.5.2, Qiskit Aer 0.17.2, x86_64, Augur commit 88a2e6a.

## What this does not show

The limits in section 7 of the pre-registration stand. These are local seconds on one machine,
two circuit families, at most 24 qubits, and a baseline that runs one method at its top setting.
A baseline that picks a middle bond dimension was not tested. Nothing here speaks to circuits too
large to solve exactly, where Augur is meant to be used.
