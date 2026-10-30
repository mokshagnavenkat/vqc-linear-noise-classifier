"""
src/metrics.py
Dual Error Mitigation Engine (Readout Error Inversion + Mitiq Zero-Noise Extrapolation)
and Quantum Circuit Evaluation Suite for Qiskit Fall Fest 2026 (PS-Q3).
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import os
import sys
import types
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Compatibility bridges for Mitiq with NumPy 2.x and Qiskit 2.x
if "numpy.lib.polynomial" not in sys.modules:
    mod = types.ModuleType("numpy.lib.polynomial")
    mod.RankWarning = np.exceptions.RankWarning
    sys.modules["numpy.lib.polynomial"] = mod

try:
    import cirq.experiments.qubit_characterizations as qc
    qc._gate_seq_to_mats = lambda *a, **k: None
except Exception:
    pass

import qiskit_aer
import qiskit.providers
qiskit.providers.aer = qiskit_aer
sys.modules["qiskit.providers.aer"] = qiskit_aer
sys.modules["qiskit.providers.aer.noise"] = qiskit_aer.noise

import qiskit.qasm2
from qiskit import QuantumCircuit, transpile
QuantumCircuit.qasm = lambda self: qiskit.qasm2.dumps(self)

from mitiq.interface.mitiq_qiskit import conversions
conversions._transform_registers = lambda circuit, **kw: circuit

from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import ExpFactory, LinearFactory

# Benchmark noise model imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "noise_model"))
from benchmark_noise_model import (
    get_benchmark_noise_model,
    prepare_for_noisy_run,
    noisy_simulator,
    P_READOUT,
    VERSION
)


# -------------------------------------------------------------------------
# 1. STRUCTURAL CIRCUIT METRICS
# -------------------------------------------------------------------------
def compute_circuit_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Calculates circuit depth, gate counts, two-qubit CX gates, and operation breakdown."""
    depth = circuit.depth()
    ops = circuit.count_ops()
    cx_count = ops.get("cx", 0)
    swap_count = ops.get("swap", 0)
    total_gates = sum(ops.values())
    return {
        "depth": depth,
        "total_gates": total_gates,
        "cx_count": cx_count,
        "swap_count": swap_count,
        "gate_breakdown": {k: int(v) for k, v in ops.items()},
    }


# -------------------------------------------------------------------------
# 2. METHOD 1: READOUT ERROR MITIGATION (REM) VIA MATRIX INVERSION
# -------------------------------------------------------------------------
def build_readout_confusion_matrix(
    num_qubits: int = 4,
    p_ro: float = P_READOUT
) -> np.ndarray:
    r"""
    Constructs the 2^n x 2^n readout confusion calibration matrix M:
    M_ij = P(measured bitstring i | prepared bitstring j)
    For symmetric uncorrelated single-qubit bit-flip error rate p:
    M = ( (1-p)  p )^{\otimes n}
        (   p  (1-p) )
    """
    m1 = np.array([[1.0 - p_ro, p_ro],
                   [p_ro, 1.0 - p_ro]], dtype=float)
    M = m1
    for _ in range(num_qubits - 1):
        M = np.kron(M, m1)
    return M


def project_to_probability_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projects arbitrary real vector v onto the probability simplex {p >= 0, sum(p) = 1}
    via exact O(N log N) Euclidean projection algorithm (Wang & Carreira-Perpinán, 2013).
    Guarantees mathematically rigorous, non-negative probability distributions.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w / np.sum(w)


def apply_readout_error_mitigation(
    counts: Dict[str, int],
    num_qubits: int = 4,
    M_inv: Optional[np.ndarray] = None,
    p_ro: float = P_READOUT
) -> Tuple[np.ndarray, float]:
    """
    Applies inverse confusion matrix unfolding and simplex projection:
    p_mitigated = SimplexProject( M^{-1} * p_measured )
    Returns:
        (mitigated_distribution_vector, mitigated_probability_qubit0_is_1)
    """
    if M_inv is None:
        M = build_readout_confusion_matrix(num_qubits, p_ro)
        M_inv = np.linalg.inv(M)

    dim = 2 ** num_qubits
    total_shots = max(1, sum(counts.values()))
    p_noisy = np.zeros(dim, dtype=float)

    for bitstr, cnt in counts.items():
        idx = int(bitstr, 2)
        if idx < dim:
            p_noisy[idx] = cnt / total_shots

    p_raw_mitigated = M_inv @ p_noisy
    p_clean = project_to_probability_simplex(p_raw_mitigated)

    # Qubit 0 is the least significant bit (odd indices in big-endian bitstrings)
    p_q0_one = float(sum(p_clean[i] for i in range(dim) if i % 2 == 1))
    return p_clean, p_q0_one


# -------------------------------------------------------------------------
# 3. METHOD 2: ZERO-NOISE EXTRAPOLATION (ZNE) VIA MITIQ GLOBAL FOLDING
# -------------------------------------------------------------------------
def apply_zne_single_circuit(
    circuit_without_measure: QuantumCircuit,
    sim: qiskit_aer.AerSimulator,
    shots: int = 8192,
    scale_factors: List[float] = [1.0, 1.5, 2.0],
    factory_type: str = "exponential"
) -> Tuple[float, Dict[str, Any]]:
    """
    Applies Zero-Noise Extrapolation using Mitiq global circuit folding across
    scale factors lambda=[1.0, 1.5, 2.0] under the official benchmark noise model.
    Transpiles folded circuits using prepare_for_noisy_run().
    Extrapolates expectation value <Z_0> to zero noise and returns mitigated P(q0=1).
    """
    scaled_expvals = []
    scaled_p1s = []

    for s in scale_factors:
        if s > 1.0:
            folded_qc = fold_global(circuit_without_measure, scale_factor=s)
        else:
            folded_qc = circuit_without_measure.copy()

        folded_qc.measure_all()
        tqc = prepare_for_noisy_run(folded_qc)
        job = sim.run(tqc, shots=shots)
        counts = job.result().get_counts()

        # Compute expectation value <Z_0>
        total = sum(counts.values())
        c0 = sum(cnt for b, cnt in counts.items() if b[-1] == "0")
        c1 = sum(cnt for b, cnt in counts.items() if b[-1] == "1")
        exp_z0 = (c0 - c1) / max(1, total)
        p1 = c1 / max(1, total)

        scaled_expvals.append(exp_z0)
        scaled_p1s.append(p1)

    # Fit and extrapolate to scale factor 0
    exp_factory = ExpFactory(scale_factors=scale_factors)
    lin_factory = LinearFactory(scale_factors=scale_factors)

    # Exponential extrapolation
    try:
        extrap_z0_exp = float(exp_factory.extrapolate(scale_factors, scaled_expvals))
    except Exception:
        extrap_z0_exp = float(lin_factory.extrapolate(scale_factors, scaled_expvals))

    # Linear extrapolation
    extrap_z0_lin = float(lin_factory.extrapolate(scale_factors, scaled_expvals))

    chosen_z0 = extrap_z0_exp if factory_type == "exponential" else extrap_z0_lin
    chosen_z0 = float(np.clip(chosen_z0, -1.0, 1.0))
    mitigated_p1 = float(np.clip((1.0 - chosen_z0) / 2.0, 0.0, 1.0))

    details = {
        "scale_factors": scale_factors,
        "scaled_expvals": [round(v, 4) for v in scaled_expvals],
        "scaled_p1s": [round(v, 4) for v in scaled_p1s],
        "extrapolated_z0_exp": round(extrap_z0_exp, 4),
        "extrapolated_z0_lin": round(extrap_z0_lin, 4),
        "mitigated_p1": round(mitigated_p1, 4),
        "unmitigated_p1": round(scaled_p1s[0], 4),
    }

    return mitigated_p1, details


# -------------------------------------------------------------------------
# 4. COMPREHENSIVE DUAL MITIGATION BENCHMARK SUITE
# -------------------------------------------------------------------------
def evaluate_noisy_and_mitigated(
    base_circuit_without_measure: QuantumCircuit,
    x_params: List[Any],
    theta_params: List[Any],
    theta: np.ndarray,
    X_angles_subset: np.ndarray,
    y_true_subset: np.ndarray,
    shots: int = 8192,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates a subset of validation samples across all operational conditions:
    1. Ideal (noiseless)
    2. Noisy unmitigated (benchmark noise model QFF2026-v2)
    3. REM Mitigated (Method 1)
    4. ZNE Mitigated (Method 2: Mitiq global folding [1, 1.5, 2] + ExpFactory)
    Returns ROC-AUC, F1, Accuracy, and mitigation details.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from qiskit_aer import AerSimulator

    ideal_sim = AerSimulator(seed_simulator=seed)
    noisy_sim = AerSimulator(noise_model=get_benchmark_noise_model(), seed_simulator=seed)

    M_inv = np.linalg.inv(build_readout_confusion_matrix(4, P_READOUT))

    p_ideal_list = []
    p_noisy_list = []
    p_rem_list = []
    p_zne_list = []

    scale_factors = [1.0, 1.5, 2.0]

    for idx, x in enumerate(X_angles_subset):
        pdict = {p: x[i] for i, p in enumerate(x_params)}
        pdict.update({p: theta[i] for i, p in enumerate(theta_params)})
        bound_qc = base_circuit_without_measure.assign_parameters(pdict)

        # 1. Ideal
        meas_qc = bound_qc.copy()
        meas_qc.measure_all()
        tqc_ideal = prepare_for_noisy_run(meas_qc)
        ideal_counts = ideal_sim.run(tqc_ideal, shots=shots).result().get_counts()
        p1_ideal = sum(cnt for b, cnt in ideal_counts.items() if b[-1] == "1") / shots
        p_ideal_list.append(p1_ideal)

        # 2. Noisy unmitigated
        tqc_noisy = prepare_for_noisy_run(meas_qc)
        noisy_counts = noisy_sim.run(tqc_noisy, shots=shots).result().get_counts()
        p1_noisy = sum(cnt for b, cnt in noisy_counts.items() if b[-1] == "1") / shots
        p_noisy_list.append(p1_noisy)

        # 3. REM Mitigated
        _, p1_rem = apply_readout_error_mitigation(noisy_counts, 4, M_inv)
        p_rem_list.append(p1_rem)

        # 4. ZNE Mitigated (Mitiq global folding across 1.0, 1.5, 2.0)
        p1_zne, _ = apply_zne_single_circuit(
            bound_qc, noisy_sim, shots=shots, scale_factors=scale_factors
        )
        p_zne_list.append(p1_zne)

    def _calc_scores(probs):
        arr = np.clip(np.array(probs), 0.0, 1.0)
        preds = (arr >= 0.5).astype(int)
        try:
            auc = round(float(roc_auc_score(y_true_subset, arr)), 4)
        except Exception:
            auc = 0.5
        f1 = round(float(f1_score(y_true_subset, preds)), 4)
        acc = round(float(accuracy_score(y_true_subset, preds)), 4)
        return {"roc_auc": auc, "f1": f1, "accuracy": acc, "scores": arr.tolist()}

    return {
        "ideal": _calc_scores(p_ideal_list),
        "noisy_unmitigated": _calc_scores(p_noisy_list),
        "rem_mitigated": _calc_scores(p_rem_list),
        "zne_mitigated": _calc_scores(p_zne_list),
        "shots_per_eval": shots,
        "n_samples": len(y_true_subset),
    }
