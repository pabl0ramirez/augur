# Augur

Augur estimates what an ideal quantum computer would measure, using classical simulators, and
tells you how far each number can be trusted.

You give it a circuit and Pauli observables (`Z0`, `Z3Z4`, `X0Y1`, weighted sums). It runs more
than one method, climbs each method up a ladder of settings until the value stops moving, and
returns every value with an empirical error bar and a label.

| Label | Meaning |
|---|---|
| `exact` | Computed exactly (statevector, small circuits) |
| `lightcone` | Exact, because only a small part of the circuit affects this observable |
| `agree` | Two independent methods converged to the same value |
| `disagree:<a>\|<b>` | Two methods converged to different values |
| `unconverged:<method>` | The ladder stopped before the value settled |
| `degenerate:<method>` | The method returned a value it cannot vouch for, such as a truncated zero |

## The name

An augur, in ancient Rome, was a priest who read the will of the gods from signs he could observe,
mostly the flight and calls of birds. He never saw the plan itself. He watched what was visible and
inferred what was hidden. Augur does the same with quantum circuits. It never runs the quantum
computer. It reads what the simulators return and infers what the ideal machine would give.

## What it does

- **Engines behind one call.** Exact statevector (up to `EXACT_MAX_QUBITS` = 24, a memory choice
  for an ordinary machine), Qiskit Aer MPS, IBM pauli-prop Pauli propagation, an exact light cone
  (Qiskit's `LightCone` pass, used when the cone has at most 20 qubits), and BlueQubit's cloud
  devices (MPS and Pauli path on CPU or GPU, run by circuit id when the circuit is hidden).
- **Ladders and convergence.** Bond dimension climbs for MPS and the truncation threshold falls for
  Pauli path. A value is converged when the last rungs (2 for MPS, 3 for Pauli path) all lie within
  `eps_tol` of each other (default 1e-2, set per run with `eps_tol` in the day spec); `sigma` is that
  max minus min spread. Set `eps_tol` below the scale of your values, or tiny values will look
  converged.
- **Fusion.** Per observable, exact beats light cone beats ladders. Converged methods are averaged
  by inverse variance. If two converged methods differ by more than 3 combined sigmas, the tighter
  one is kept, sigma becomes the gap, and the value is labelled `disagree`.
- **Calibration.** On circuits with known answers, `bench` scores each method and `calibrate` gives
  each a trust weight of 1 / (rmse + 1e-3), normalised so the best method is 1. Trust divides a
  method's sigma in fusion and scales the gain the campaign expects from its next run.
- **A budgeted campaign.** Across many circuits, the next run is the one with the highest expected
  gain per predicted cost, within a budget and a deadline. Cloud runs respect a provider's job cap,
  resubmit jobs the cap bounced, and drop requests the server refused. With `--resume`, every rung
  already in the run log is restored and not bought again (`test_resume_buys_nothing_twice`).
- **A watchdog** that restarts a dead run with `--resume` and never restarts a live one
  (`test_watch_never_restarts_a_live_run_even_when_quiet`), and alerts by email if you configure
  SMTP. A failed or unconfigured email is recorded, never fatal.
- **A self-check** (`augur pulse`) that tests every engine against exact values on a small circuit.

## Install and try

    pip install -e ".[test]"          # add ".[cloud]" for BlueQubit
    python -m pytest -q tests/        # 217 tests
    augur pulse                       # self-check against exact values
    augur estimate --qasm circuit.qasm --obs Z0,Z1Z2,X5
    augur bench                       # score each method on practice circuits

For the cloud campaign, set `BLUEQUBIT_API_TOKEN`, write a day spec (circuits, observables,
budget, deadline, per-device ladders) and run `augur day --spec spec.json --confirm-spend`.
Nothing that spends money runs without `--confirm-spend`.

## Where it has been used

At the BlueQubit x DuQuantum quantum echoes hackathon (Duke University, 24 and 25 October 2026),
Augur drove most graded runs. Each problem was scored by the organizers as the squared correlation
between our values and their exact values, times the problem's points. Its bond-dimension ladders
took one 36-qubit chain problem from 426 to 564 out of 600 and a 30-qubit chain problem from 157
to 235 out of 300. On a deep near-Clifford grid and a 38-qubit random grid neither method found
signal within the budget, and Augur's labels said so rather than passing off a truncated zero as
an answer. Scores, receipts and the full write-up are in
[duke-otoc-hackathon](https://github.com/pabl0ramirez/duke-otoc-hackathon).

## Limits

- A converged ladder is evidence, not proof. `sigma` is the spread between the last rungs, an
  empirical error bar, never a bound.
- Deep, strongly scrambling circuits can defeat both MPS and Pauli path at affordable settings.
- The watchdog identifies a run through `/proc`, so it is exact on Linux. On macOS any process
  with the run's PID counts as alive, which can delay a restart but never starts a second paid
  run. It refuses to run on Windows.
- Augur makes no savings claim. A pre-registered test against running every circuit once at the
  top MPS setting found none at 18 or 24 qubits. At 18 qubits the top setting was faster on every
  repeat. At 24 qubits Augur reached the target on all 12 circuits where the top setting reached
  7, but its median cost was lower in only two repeats of the three the rule required. Results
  and the command to check them are in [benchmarks/COST_TEST_RESULT.md](benchmarks/COST_TEST_RESULT.md).
- A seconds budget is not a hard cap. A local rung is not stopped once it starts, and in that test
  one campaign spent 17% over its budget.

## Prior art

See [PRIOR_ART.md](PRIOR_ART.md) for the related work we searched and how Augur differs.

## License

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may read, run,
modify and share it for noncommercial purposes, including research and education. Commercial use,
including deployment inside a company, needs a separate license; ask Matrix CR Studio at
pablo@matrixcr.ai before adopting it. Augur's dependencies keep their own licenses (Apache-2.0
for Qiskit and Qiskit Aer) and are installed from PyPI, not redistributed here.

Copyright 2026 Pablo Ramirez, Matrix CR Studio, Costa Rica.
