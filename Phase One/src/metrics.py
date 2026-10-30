"""
src/metrics.py
Circuit fidelity estimators, calibrated Zero-Noise Extrapolation (ZNE scale factors [1.0, 1.3, 1.7]),
Readout Error Inversion (REM) with Euclidean simplex projection, and shot-scaled (N=8192)
error mitigation benchmark suite for Qiskit Fall Fest 2026.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Tuple, Any, Optional, Callable
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# NumPy 2 / Cirq compatibility bridge for Mitiq
import types
if "numpy.lib.polynomial" not in sys.modules:
    mod = types.ModuleType("numpy.lib.polynomial")
    mod.RankWarning = np.exceptions.RankWarning
    sys.modules["numpy.lib.polynomial"] = mod
try:
    import cirq.experiments.qubit_characterizations as qc
    qc._gate_seq_to_mats = lambda *a, **k: None
except Exception:
    pass

try:
    import mitiq
    from mitiq.zne.scaling import fold_gates_at_random
    from mitiq.zne.inference import RichardsonFactory, LinearFactory, PolyFactory, ExpFactory
    ExponentialFactory = ExpFactory
    MITIQ_AVAILABLE = True
except Exception:
    MITIQ_AVAILABLE = False
    ExponentialFactory = None

from src.processors import (
    get_unified_noise_model,
    get_qiskit_coupling_map,
    get_aer_backend
)
from src.problem import (
    build_vqc_circuit,
    bind_vqc_parameters,
    load_processed_data
)


# -------------------------------------------------------------------------
# STRUCTURAL CIRCUIT METRICS & ESTIMATED FIDELITY
# -------------------------------------------------------------------------
def compute_circuit_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Computes depth, total gates, CX count, and gate breakdown."""
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
        "gate_breakdown": {k: int(v) for k, v in ops.items()}
    }


def estimate_circuit_fidelity(
    circuit: QuantumCircuit,
    gate_error_1q: float = 0.001,
    gate_error_2q: float = 0.005,
    readout_error: float = 0.01
) -> float:
    """
    Estimates overall circuit execution fidelity F via the Markovian independent error model:
    F ≈ (1 - e_1q)^{N_1q} * (1 - e_2q)^{N_2q} * (1 - e_ro)^{n_qubits}
    """
    ops = circuit.count_ops()
    n_2q = ops.get("cx", 0) + ops.get("cz", 0) + ops.get("swap", 0) * 3
    n_1q = sum(v for k, v in ops.items() if k not in ["cx", "cz", "swap", "barrier", "measure"])
    n_qubits = circuit.num_qubits
    
    fid_1q = (1.0 - gate_error_1q) ** n_1q
    fid_2q = (1.0 - gate_error_2q) ** n_2q
    fid_ro = (1.0 - readout_error) ** n_qubits
    
    return float(np.clip(fid_1q * fid_2q * fid_ro, 0.0, 1.0))


# -------------------------------------------------------------------------
# CALIBRATED ZERO-NOISE EXTRAPOLATION (TIGHT SCALE FACTORS: 1.0, 1.3, 1.7)
# -------------------------------------------------------------------------
def fold_quantum_circuit(circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    """
    Applies unitary folding to amplify two-qubit noise by exact scale factor lambda:
    - scale_factor = 1.0: Unchanged original circuit.
    - scale_factor in (1.0, 3.0]: Evenly inserts identity CX pairs (CX * CX = I)
      across the two-qubit gates, strictly scaling the effective entangling error rate
      while preserving the underlying logical unitary operation.
    """
    if scale_factor <= 1.0:
        return circuit.copy()
        
    cx_indices = [i for i, inst in enumerate(circuit.data) if inst.operation.name == "cx"]
    m = len(cx_indices)
    if m == 0:
        return circuit.copy()
        
    # Total CX pairs needed to achieve scale_factor
    num_pairs_total = int(round((scale_factor - 1.0) * m / 2.0))
    pairs_per_cx = np.zeros(m, dtype=int)
    for p in range(num_pairs_total):
        pairs_per_cx[p % m] += 1
        
    folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name=f"{circuit.name}_folded_{scale_factor:.1f}")
    cx_count = 0
    for inst in circuit.data:
        op = inst.operation
        q = inst.qubits
        c = inst.clbits
        folded.append(op, q, c)
        if op.name == "cx":
            for _ in range(pairs_per_cx[cx_count]):
                folded.cx(q[0], q[1])
                folded.cx(q[0], q[1])
            cx_count += 1
            
    return folded


def linear_factory_extrapolate(scale_factors: List[float], values: List[float]) -> float:
    """
    Tightly-optimized linear zero-noise extrapolation using Mitiq LinearFactory
    (with exact analytical linear regression fallback).
    Minimizes shot-noise amplification variance across calibrated scale bounds [1.0, 1.3, 1.7].
    """
    if MITIQ_AVAILABLE:
        try:
            lf = LinearFactory(scale_factors=scale_factors)
            return float(lf.extrapolate(scale_factors, values))
        except Exception:
            pass
    poly = np.polyfit(scale_factors, values, deg=1)
    return float(poly[-1])


def exponential_factory_extrapolate(scale_factors: List[float], values: List[float]) -> float:
    """
    Exponential zero-noise extrapolation using Mitiq ExpFactory (or scipy curve_fit).
    Fits E(lambda) = a * exp(-b * lambda) back to lambda = 0.
    """
    if MITIQ_AVAILABLE:
        try:
            ef = ExpFactory(scale_factors=scale_factors)
            return float(ef.extrapolate(scale_factors, values))
        except Exception:
            pass
    try:
        def exp_model(x, a, b):
            return a * np.exp(-b * x)
        popt, _ = curve_fit(exp_model, scale_factors, values, p0=[values[0], 0.1], maxfev=1000)
        return float(popt[0])
    except Exception:
        return linear_factory_extrapolate(scale_factors, values)


def richardson_extrapolate(scale_factors: List[float], values: List[float]) -> float:
    """
    Analytic Richardson Extrapolation to noise scale lambda = 0:
    - For [1.0, 2.0, 3.0]: Exact weights w = [3.0, -3.0, 1.0] (sum=1.0, Var ampl=19)
    - For [1.0, 1.5, 2.0]: Exact weights w = [6.0, -8.0, 3.0] (sum=1.0)
    - Generalized: Polynomial Lagrange interpolation at lambda = 0.
    """
    s = np.array(scale_factors, dtype=float)
    v = np.array(values, dtype=float)
    
    if len(s) == 3 and np.allclose(s, [1.0, 2.0, 3.0]):
        val_0 = 3.0 * v[0] - 3.0 * v[1] + 1.0 * v[2]
        return float(val_0)
        
    if len(s) == 3 and np.allclose(s, [1.0, 1.5, 2.0]):
        val_0 = 6.0 * v[0] - 8.0 * v[1] + 3.0 * v[2]
        return float(val_0)
        
    if len(s) == 2:
        val_0 = (s[1] * v[0] - s[0] * v[1]) / (s[1] - s[0])
        return float(val_0)
        
    poly = np.polyfit(s, v, deg=len(s) - 1)
    return float(poly[-1])


def polynomial_extrapolate(scale_factors: List[float], values: List[float], degree: int = 1) -> float:
    """Polynomial regression (default linear degree=1 for minimum shot-noise amplification)."""
    deg = min(degree, len(scale_factors) - 1)
    poly = np.polyfit(scale_factors, values, deg=deg)
    return float(poly[-1])


def execute_circuit_with_zne(
    circuit: QuantumCircuit,
    backend: AerSimulator,
    shots: int = 8192,
    scale_factors: List[float] = [1.0, 1.3, 1.7],
    factory: str = "linear",
    observable_fn: Optional[Callable[[Dict[str, int]], float]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Executes circuit at noise scale factors [1.0, 1.3, 1.7] and extrapolates to zero noise
    using Mitiq LinearFactory / ExponentialFactory.
    Observable defaults to expectation value of Pauli Z on qubit 0: <Z_0>.
    """
    if observable_fn is None:
        def observable_fn(counts: Dict[str, int]) -> float:
            total = sum(counts.values())
            c0 = sum(cnt for bitstr, cnt in counts.items() if bitstr[-1] == "0")
            c1 = sum(cnt for bitstr, cnt in counts.items() if bitstr[-1] == "1")
            return (c0 - c1) / max(1, total)

    scale_values = []
    folded_circuits = []
    
    for s in scale_factors:
        f_qc = fold_quantum_circuit(circuit, scale_factor=s)
        folded_circuits.append(f_qc)
        
    job = backend.run(folded_circuits, shots=shots)
    results = job.result()
    
    for i, s in enumerate(scale_factors):
        counts = results.get_counts(i)
        ev = observable_fn(counts)
        scale_values.append(ev)
        
    linear_val = linear_factory_extrapolate(scale_factors, scale_values)
    exp_val = exponential_factory_extrapolate(scale_factors, scale_values)
    richardson_val = richardson_extrapolate(scale_factors, scale_values)
    
    chosen_val = exp_val if factory == "exponential" else linear_val
    
    zne_details = {
        "scale_factors": scale_factors,
        "scaled_expectation_values": [round(v, 5) for v in scale_values],
        "linear_factory_zero_noise": round(linear_val, 5),
        "exponential_factory_zero_noise": round(exp_val, 5),
        "richardson_zero_noise": round(richardson_val, 5),
        "extrapolated_zero_noise": round(chosen_val, 5),
        "unmitigated_noisy_val": round(scale_values[0], 5),
        "factory_used": factory,
        "mitiq_used": MITIQ_AVAILABLE
    }
    
    return chosen_val, zne_details


# -------------------------------------------------------------------------
# READOUT ERROR INVERSION (REM) WITH SIMPLEX PROJECTION
# -------------------------------------------------------------------------
def build_readout_confusion_matrix(num_qubits: int = 4, error_rate: float = 0.01) -> np.ndarray:
    r"""
    Constructs the 2^n x 2^n readout confusion calibration matrix M:
    M_ij = P(measured bitstring i | prepared bitstring j)
    For symmetric uncorrelated single-qubit bit-flip error rate p:
    M = ( (1-p)  p )^{\otimes n}
        (   p  (1-p) )
    """
    m1 = np.array([[1.0 - error_rate, error_rate],
                   [error_rate, 1.0 - error_rate]], dtype=float)
    M = m1
    for _ in range(num_qubits - 1):
        M = np.kron(M, m1)
    return M


def project_to_probability_simplex(v: np.ndarray) -> np.ndarray:
    """
    Projects arbitrary real vector v onto the probability simplex {p >= 0, sum(p) = 1}
    via exact O(N log N) Euclidean projection algorithm (Wang & Carreira-Perpinán, 2013).
    Guarantees non-negative, normalized probability distributions after matrix inversion.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w / np.sum(w)


def apply_readout_error_inversion(
    counts: Dict[str, int],
    num_qubits: int = 4,
    readout_error_rate: float = 0.01,
    M_inv: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Applies inverse confusion matrix unfolding and simplex projection:
    p_mitigated = SimplexProject( M^{-1} * p_measured )
    """
    if M_inv is None:
        M = build_readout_confusion_matrix(num_qubits, readout_error_rate)
        M_inv = np.linalg.inv(M)
        
    dim = 2 ** num_qubits
    total_shots = sum(counts.values())
    p_noisy = np.zeros(dim, dtype=float)
    
    for bitstr, cnt in counts.items():
        idx = int(bitstr, 2)
        if idx < dim:
            p_noisy[idx] = cnt / total_shots
            
    # Matrix inversion
    p_raw_mitigated = M_inv @ p_noisy
    
    # Simplex projection
    p_clean = project_to_probability_simplex(p_raw_mitigated)
    
    # Reconstruct count-like probability dictionary
    mitigated_counts = {}
    fmt = f"0{num_qubits}b"
    for idx, prob in enumerate(p_clean):
        if prob > 1e-6:
            mitigated_counts[format(idx, fmt)] = float(prob)
            
    return mitigated_counts


def expectation_value_from_probs(
    prob_dict: Dict[str, float],
    qubit_index: int = 0
) -> float:
    """Computes Pauli Z expectation value <Z_i> from a probability dictionary."""
    ev = 0.0
    for bitstr, prob in prob_dict.items():
        char = bitstr[-(qubit_index + 1)]
        sign = 1.0 if char == "0" else -1.0
        ev += sign * prob
    return float(ev)


# -------------------------------------------------------------------------
# END-TO-END VQC EVALUATION PIPELINE (SCALED SHOTS N=8192)
# -------------------------------------------------------------------------
def evaluate_vqc_predictions(
    circuit_template: QuantumCircuit,
    theta: np.ndarray,
    X_data: np.ndarray,
    backend: AerSimulator,
    processor_name: str,
    mitigation: str = "none",  # "none", "rem", "zne", "both"
    shots: int = 8192,
    scale_factors: List[float] = [1.0, 1.3, 1.7],
    factory: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates predictions and classification accuracy across a dataset:
    - Transpiles circuit template for processor with SABRE layout & routing
    - Binds parameters for each sample
    - Runs circuit under simulator with N=8192 shots
    - Applies selected mitigation (None, REM, ZNE [1.0, 1.3, 1.7], or Both)
    - Returns predicted classes (0 or 1) and continuous probabilities
    """
    coupling_map = get_qiskit_coupling_map(processor_name)
    transpiled_template = transpile(
        circuit_template,
        coupling_map=coupling_map,
        basis_gates=["sx", "x", "rz", "cx", "id"],
        optimization_level=3,
        routing_method="sabre",
        seed_transpiler=42
    )
    
    M_inv = None
    if mitigation in ["rem", "both"]:
        M = build_readout_confusion_matrix(num_qubits=4, error_rate=0.01)
        M_inv = np.linalg.inv(M)
        
    y_preds = []
    y_probs = []
    
    # Process batch
    bound_circuits = []
    for x_sample in X_data:
        qc_sample = bind_vqc_parameters(transpiled_template.copy(), x_sample, theta)
        bound_circuits.append(qc_sample)
        
    if mitigation in ["none", "rem"]:
        job = backend.run(bound_circuits, shots=shots)
        results = job.result()
        for i in range(len(X_data)):
            counts = results.get_counts(i)
            if mitigation == "rem":
                probs = apply_readout_error_inversion(counts, num_qubits=4, M_inv=M_inv)
                ev = expectation_value_from_probs(probs, qubit_index=0)
            else:
                total = sum(counts.values())
                c0 = sum(cnt for b, cnt in counts.items() if b[-1] == "0")
                c1 = sum(cnt for b, cnt in counts.items() if b[-1] == "1")
                ev = (c0 - c1) / max(1, total)
                
            prob_class_1 = float(np.clip((1.0 - ev) / 2.0, 0.0, 1.0))
            pred = 1 if prob_class_1 >= 0.5 else 0
            y_probs.append(prob_class_1)
            y_preds.append(pred)
            
    elif mitigation in ["zne", "both"]:
        for qc_sample in bound_circuits:
            folded_circs = [fold_quantum_circuit(qc_sample, s) for s in scale_factors]
            job = backend.run(folded_circs, shots=shots)
            res = job.result()
            
            evs = []
            for s_idx in range(len(scale_factors)):
                cnts = res.get_counts(s_idx)
                if mitigation == "both":
                    p_dict = apply_readout_error_inversion(cnts, num_qubits=4, M_inv=M_inv)
                    ev_s = expectation_value_from_probs(p_dict, qubit_index=0)
                else:
                    tot = sum(cnts.values())
                    c0 = sum(cnt for b, cnt in cnts.items() if b[-1] == "0")
                    c1 = sum(cnt for b, cnt in cnts.items() if b[-1] == "1")
                    ev_s = (c0 - c1) / max(1, tot)
                evs.append(ev_s)
                
            if factory == "exponential":
                ev_zero = exponential_factory_extrapolate(scale_factors, evs)
            elif factory == "richardson":
                ev_zero = richardson_extrapolate(scale_factors, evs)
            else:
                ev_zero = linear_factory_extrapolate(scale_factors, evs)
                
            prob_class_1 = float(np.clip((1.0 - ev_zero) / 2.0, 0.0, 1.0))
            pred = 1 if prob_class_1 >= 0.5 else 0
            y_probs.append(prob_class_1)
            y_preds.append(pred)
            
    return np.array(y_preds), np.array(y_probs)


def run_mitigation_benchmark_matrix(
    output_dir: str = "results/tables",
    shots: int = 8192,
    scale_factors: List[float] = [1.0, 1.3, 1.7],
    factory: str = "linear"
) -> pd.DataFrame:
    """
    Executes the definitive mitigation benchmark sweep:
    Processors: Processor A (5Q), Processor B (9Q)
    Regimes:
      1. Ideal (Noise-free Statevector, N=8192)
      2. Noisy Unmitigated (Unified Noise Model, N=8192)
      3. REM (Readout Error Inversion, N=8192)
      4. ZNE (Zero-Noise Extrapolation: Scale [1.0, 1.3, 1.7], N=8192)
      5. Full Hybrid Mitigated (REM + ZNE, N=8192)
    Computes accuracy, error rate, mitigation gain, and fidelity metrics.
    Saves tables to results/tables/mitigation_benchmarks.csv and .json.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed_data()
    vqc_template = build_vqc_circuit(num_qubits=4, reps=2, entanglement="sca")
    
    # Load optimal theta if available, else train
    theta_path = os.path.join("data/processed", "optimal_theta.npy")
    if os.path.exists(theta_path):
        theta = np.load(theta_path)
    else:
        from src.problem import train_vqc
        theta, _ = train_vqc(max_iter=150, seed=42)
        
    processors = ["A", "B"]
    mitigation_strategies = ["none", "rem", "zne", "both"]
    records = []
    
    # 1. Ideal baseline (Noise-Free, N=8192)
    ideal_backend = AerSimulator(method="statevector")
    y_pred_ideal, _ = evaluate_vqc_predictions(
        vqc_template, theta, X_test, ideal_backend, "A", mitigation="none", shots=shots
    )
    acc_ideal = float(np.mean(y_pred_ideal == y_test))
    
    records.append({
        "processor": "Ideal_Statevector",
        "regime": "Ideal (Noise-Free)",
        "mitigation": "none",
        "accuracy": round(acc_ideal * 100.0, 2),
        "error_rate": round((1.0 - acc_ideal) * 100.0, 2),
        "mitigation_gain": 0.0,
        "estimated_fidelity": 1.000,
        "circuit_depth": vqc_template.depth(),
        "cx_count": vqc_template.count_ops().get("cx", 0)
    })
    
    print(f"Ideal VQC Baseline Accuracy (N={shots}): {acc_ideal * 100.0:.2f}%")
    
    # 2. Benchmark noisy and mitigated regimes per processor
    for proc in processors:
        # Determine transpiled circuit specs
        cmap = get_qiskit_coupling_map(proc)
        tqc = transpile(
            vqc_template,
            coupling_map=cmap,
            basis_gates=["sx", "x", "rz", "cx", "id"],
            optimization_level=3,
            routing_method="sabre",
            seed_transpiler=42
        )
        proc_depth = tqc.depth()
        proc_cx = tqc.count_ops().get("cx", 0)
        proc_fid = estimate_circuit_fidelity(tqc)
        
        base_noisy_acc = None
        
        for strat in mitigation_strategies:
            backend_strat = get_aer_backend(proc, noisy=True, seed_simulator=42)
            y_pred, _ = evaluate_vqc_predictions(
                vqc_template,
                theta,
                X_test,
                backend_strat,
                proc,
                mitigation=strat,
                shots=shots,
                scale_factors=scale_factors,
                factory=factory
            )
            acc = float(np.mean(y_pred == y_test))
            
            if strat == "none":
                base_noisy_acc = acc
                regime_name = "Noisy Unmitigated"
            elif strat == "rem":
                regime_name = "Readout Inversion (REM)"
            elif strat == "zne":
                regime_name = "Zero-Noise Extrapolation (ZNE)"
            elif strat == "both":
                regime_name = "Full Hybrid (ZNE + REM)"
                
            gain = round((acc - base_noisy_acc) * 100.0, 2)
            
            records.append({
                "processor": f"Processor_{proc}",
                "regime": regime_name,
                "mitigation": strat,
                "accuracy": round(acc * 100.0, 2),
                "error_rate": round((1.0 - acc) * 100.0, 2),
                "mitigation_gain": gain,
                "estimated_fidelity": round(proc_fid, 4),
                "circuit_depth": proc_depth,
                "cx_count": proc_cx
            })
            print(f"[{proc}] {regime_name}: Acc={acc*100.0:.2f}%, Gain={gain:+.2f}%")
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "mitigation_benchmarks.csv")
    json_path = os.path.join(output_dir, "mitigation_benchmarks.json")
    df.to_csv(csv_path, index=False)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    print(f"Mitigation benchmark matrix saved to {csv_path} and {json_path}")
    return df


if __name__ == "__main__":
    df = run_mitigation_benchmark_matrix()
    print("\nMitigation Benchmark Summary:")
    print(df.to_string(index=False))
