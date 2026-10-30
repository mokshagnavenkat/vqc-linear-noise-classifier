# QFF2026 QML Track — Results Report

**Team name:**
**Selected PS:** [ ] PS-Q1  [ ] PS-Q2  [ ] PS-Q3
**Date:**
**Dependency versions:** (paste `pip freeze | grep -iE 'pennylane|qiskit|scikit|numpy|pandas'`)
**Noise model version:** QFF2026-v2

---

## 1. Approach Summary
*3–5 sentences. Encoding strategy, circuit architecture, why.*

---

## 2. Results

### Classical baseline
| Model | Official metric | Notes |
|-------|----------------|-------|
| | | |

### Quantum / hybrid model
Official metric: Q1 macro-F1 · Q2 balanced accuracy · Q3 ROC-AUC (from predicted probabilities).

| Configuration | Ideal | Noisy (benchmark) | Notes |
|---------------|-------|--------------------|-------|
| | | | |

### Noise mitigation (PS-Q3 required; optional for PS-Q1/Q2)
| Strategy | Mitigated metric | Extra circuit evals | Justified? |
|----------|-----------------|---------------------|-----------|
| | | | |

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py` (8192 shots per basis).

**Fidelity definition used:** F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩) / 4, with ⟨P⟩ = (N_even − N_odd)/N, SE(⟨P⟩) = √((1 − ⟨P⟩²)/N) and SE(F) = √(SE_XX² + SE_YY² + SE_ZZ²)/4. (P(00)+P(11) is a population, not a fidelity, and is not accepted.)

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
|-----------|-----------|-----------|-----------|--------|
| Ideal | | | | |
| Noisy (benchmark) | | | | |

*What the difference says about noise and entanglement (3–5 sentences):*

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json`.

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs |j⟩ of F_j = ⟨ψ_j|ρ_j|ψ_j⟩, where ψ_j = QFT|j⟩ (ideal) and ρ_j is the density matrix of the 3 logical qubits after the transpiled circuit runs under the benchmark noise model (gate noise; no measurement). Ideal must be 1.0.

**Processor ID:**  **n_qubits:**  **Topology:**

| Benchmark | Value |
|-----------|-------|
| QFT-3 depth (before transpile) | |
| QFT-3 depth (after transpile) | |
| Two-qubit gate count | |
| SWAP overhead | |
| Fidelity (ideal) | |
| Fidelity (noisy) | |
| Coupling map direction (one-way / two-way links) | |

*What do the results tell you about your topology? (2–4 sentences)*

---

## 5. Resource Costs
| Resource | Value |
|----------|-------|
| Qubits used | |
| Circuit depth | |
| Trainable parameters | |
| Training time (approx) | |
| Total circuit evaluations | |

---

## 6. Limitations and Honest Assessment
*What didn't work? Is quantum competitive with classical here? Why or why not?*

---

## 7. Reproducibility
```python
SEED = ...
```

---

## 8. Optional Extensions
[ ] Second mitigation strategy (PS-Q3)
[ ] Ablation studies
[ ] Other:
