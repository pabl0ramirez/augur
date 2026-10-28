# Prior art

Searched on 27 October 2026 before publishing, so the difference claimed in the README can be
checked. Abstracts and repository descriptions were read, not every full paper, so this is a
record of what we found, not a proof that nothing else exists.

## What exists

| Work | What it does | Difference from Augur |
|---|---|---|
| [Gharibyan et al., arXiv 2507.10771](https://arxiv.org/abs/2507.10771) | Pauli-path simulation run as a ladder of truncation thresholds, convergence judged per run | One method, one observable at a time, no budget across circuits |
| [Rudolph et al., arXiv 2505.21606](https://arxiv.org/pdf/2505.21606), [PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl) | A computational framework and library for Pauli propagation | A simulation engine; Augur uses an engine of this kind as one of its methods |
| [IBM pauli-prop](https://quantum.cloud.ibm.com/docs/en/addons/pauli-prop.md) | A Pauli propagation engine | No convergence or budget logic |
| [Maestro, arXiv 2512.04216](https://arxiv.org/abs/2512.04216) | One interface to several simulators, choosing one by predicted runtime | Chooses for speed; no ladders, fusion, calibration or accuracy-driven budget |
| [Qiskit Aer automatic method](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) | Picks statevector or MPS from the circuit | No Pauli path, no budget logic |
| [Adaptive multi-backend near-Clifford simulation, arXiv 2607.27075](https://arxiv.org/html/2607.27075v1) | Picks a backend per sub-circuit with a cost model and splits an error budget across sub-circuits | Allocates inside one circuit; Augur allocates across many circuits by expected gain in correlation |
| Multi-fidelity allocation, e.g. [MF-Box](https://arxiv.org/pdf/2306.03144), [adaptive scale-up](https://arxiv.org/html/2404.00053v2) | Ranks the next simulation by expected gain divided by cost | The allocation principle Augur uses is this classic idea; Augur applies it to observable estimation |
| [Benchmarking zero-setup simulators, arXiv 2607.09882](https://arxiv.org/pdf/2607.09882) | Benchmarks statevector, MPS and Pauli path services | Compares simulators; does not choose among them |
| [classical-otoc-estimation](https://github.com/yusufkorkmazyigit/classical-otoc-estimation) | Another participant's code from the same hackathon | Hackathon scripts |

## What we did not find

A tool that, for observable estimation, runs two independent methods up convergence ladders,
fuses them per observable into a value with an empirical error bar and a trust label, calibrates
how much to trust each method on circuits with known answers, and allocates one budget across many
circuits by expected gain. Each piece has relatives above; we found no combination.

## Searches run

| Where | Query |
|---|---|
| Web | adaptive budget allocation simulator selection estimate quantum observables MPS Pauli propagation convergence automatic method choice 2025 2026 |
| Web | automatic simulator backend selection tensor network vs statevector vs Pauli propagation cost optimizer quantum circuits tool |
| Web | "Pauli propagation" "tensor network" cross-check fused estimate error bar quantum circuit observables classical simulation framework |
| Web | quantum circuit simulation campaign allocate compute budget across many circuits expected accuracy gain adaptive refinement multi-fidelity |
| GitHub repositories | pauli propagation, pauli path simulator, otoc simulation, expectation value estimation mps, plus four longer phrasings that returned nothing |
| GitHub code | pauli_prop matrix_product_state (no results) |
