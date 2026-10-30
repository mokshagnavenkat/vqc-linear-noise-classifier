"""
src/problem.py
Synthetic classification problem formulation, feature preprocessing,
upgraded non-linear entangling feature maps, and SPSA-optimized
Variational Quantum Classifier (VQC) ansatz construction.
"""

from typing import Tuple, Dict, Any, Optional, List
import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import real_amplitudes, RealAmplitudes, zz_feature_map, ZZFeatureMap

# -------------------------------------------------------------------------
# DATASET GENERATION & PREPROCESSING (Strict 200 samples, 4 features)
# -------------------------------------------------------------------------
def generate_and_save_data(
    n_samples: int = 200,
    n_features: int = 4,
    n_informative: int = 3,
    n_redundant: int = 0,
    random_state: int = 42,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    test_size: float = 0.20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes a 200-sample, 4-feature binary classification dataset.
    Saves raw CSV and normalized train/test matrices to data/processed/.
    Features are scaled to [0, pi] for quantum angle and phase encoding.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Synthesize classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        class_sep=1.2,
        random_state=random_state
    )
    
    # Save raw dataset
    raw_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    raw_df["target"] = y
    raw_csv_path = os.path.join(raw_dir, "synthetic_classification_200x4.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    
    # 2. Train-test split (80-20: 160 train, 40 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Angle & phase scaling: map features to [0, pi]
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed tensors
    processed_npz_path = os.path.join(processed_dir, "processed_dataset.npz")
    np.savez_compressed(
        processed_npz_path,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_
    )
    
    # Metadata JSON
    meta = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_range": [0.0, float(np.pi)],
        "class_balance_train": {int(c): int(cnt) for c, cnt in zip(*np.unique(y_train, return_counts=True))},
        "class_balance_test": {int(c): int(cnt) for c, cnt in zip(*np.unique(y_test, return_counts=True))}
    }
    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_processed_data(processed_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads cached processed train/test arrays or regenerates if missing."""
    npz_path = os.path.join(processed_dir, "processed_dataset.npz")
    if not os.path.exists(npz_path):
        return generate_and_save_data()
    data = np.load(npz_path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


# -------------------------------------------------------------------------
# UPGRADED NON-LINEAR FEATURE MAPS & ANSATZ CONSTRUCTION
# -------------------------------------------------------------------------
def build_feature_map(
    num_qubits: int = 4,
    parameter_name: str = "x",
    map_type: str = "ry_rz_entangled"
) -> QuantumCircuit:
    """
    Constructs the feature encoding map:
    - 'ry_rz_entangled' (default): 2-layer Ry-Rz entangling circuit map.
      Forces input features into higher-dimensional Hilbert space via:
      Layer 1: Ry(x_i) on each qubit
      Entangling layer: Circular CX ring (0->1, 1->2, 2->3, 3->0)
      Layer 2: Rz(x_i) on each qubit
    - 'zz': Qiskit non-linear ZZFeatureMap with reps=1.
    - 'angle': Baseline single-layer Ry angle encoding.
    """
    if map_type == "ry_rz_entangled":
        x = ParameterVector(parameter_name, length=num_qubits)
        qc = QuantumCircuit(num_qubits, name="RyRz_Entangled_FeatureMap")
        # Layer 1: Ry rotations
        for i in range(num_qubits):
            qc.ry(x[i], i)
        # Entangling ring
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        if num_qubits > 2:
            qc.cx(num_qubits - 1, 0)
        # Layer 2: Rz rotations
        for i in range(num_qubits):
            qc.rz(x[i], i)
        return qc
    elif map_type == "zz":
        try:
            return zz_feature_map(num_qubits=num_qubits, reps=1, parameter_prefix=parameter_name).decompose()
        except Exception:
            return ZZFeatureMap(num_qubits=num_qubits, reps=1, parameter_prefix=parameter_name).decompose()
    elif map_type == "angle":
        x = ParameterVector(parameter_name, length=num_qubits)
        qc = QuantumCircuit(num_qubits, name="AngleEncoding")
        for i in range(num_qubits):
            qc.ry(x[i], i)
        return qc
    else:
        raise ValueError(f"Unknown feature map type: {map_type}")


def build_angle_feature_map(num_qubits: int = 4, parameter_name: str = "x") -> QuantumCircuit:
    """Backward-compatible alias for build_feature_map with non-linear entangling default."""
    return build_feature_map(num_qubits=num_qubits, parameter_name=parameter_name, map_type="ry_rz_entangled")


def build_ansatz(
    num_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "sca"
) -> QuantumCircuit:
    """
    Constructs RealAmplitudes variational ansatz with reps=2 and shifted circular alternating ('sca')
    entanglement matching the dual-hub heavy-hex connectivity graph of Processor B.
    Number of variational parameters = num_qubits * (reps + 1) = 4 * 3 = 12 parameters.
    """
    try:
        ansatz = real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement, parameter_prefix="theta")
    except Exception:
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement, parameter_prefix="theta")
    return ansatz.decompose()


def build_vqc_circuit(
    num_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "sca",
    feature_map_type: str = "ry_rz_entangled",
    add_measurement: bool = True
) -> QuantumCircuit:
    """
    Assembles the full end-to-end VQC circuit:
    Non-linear Entangling Feature Map + RealAmplitudes Ansatz (sca/circular) + (optional) Measurement.
    """
    feature_map = build_feature_map(num_qubits=num_qubits, parameter_name="x", map_type=feature_map_type)
    ansatz = build_ansatz(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    
    # Compose circuit
    vqc = QuantumCircuit(num_qubits, num_qubits, name="VQC_Full")
    vqc.compose(feature_map, inplace=True)
    vqc.barrier()
    vqc.compose(ansatz, inplace=True)
    
    if add_measurement:
        vqc.barrier()
        vqc.measure(range(num_qubits), range(num_qubits))
        
    return vqc


def bind_vqc_parameters(
    circuit: QuantumCircuit,
    feature_values: np.ndarray,
    theta_values: np.ndarray
) -> QuantumCircuit:
    """
    Binds both classical feature parameters (x) and variational parameters (theta).
    """
    param_dict = {}
    
    # Feature vector params
    x_params = sorted([p for p in circuit.parameters if p.name.startswith("x")], key=lambda p: p.name)
    for p, val in zip(x_params, feature_values):
        param_dict[p] = float(val)
        
    # Variational params
    theta_params = sorted(
        [p for p in circuit.parameters if p not in x_params],
        key=lambda p: int(p.name.split("[")[1].split("]")[0]) if "[" in p.name and p.name.split("[")[1].split("]")[0].isdigit() else p.name
    )
    for p, val in zip(theta_params, theta_values):
        param_dict[p] = float(val)
        
    return circuit.assign_parameters(param_dict)


# -------------------------------------------------------------------------
# SPSA OPTIMIZATION LOOP (Strict limit: 150 iterations)
# -------------------------------------------------------------------------
def train_vqc(
    num_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "sca",
    feature_map_type: str = "ry_rz_entangled",
    max_iter: int = 150,
    batch_size: int = 40,
    shots: int = 512,
    seed: int = 42,
    output_dir: str = "data/processed"
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Trains the VQC RealAmplitudes ansatz parameters on the synthetic classification task
    using SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer (150 iterations limit).
    SPSA stochastic gradients navigate local minima induced by noise and parameter landscapes.
    Logs iteration loss and mean energy trajectory, saving optimal theta to data/processed/optimal_theta.npy.
    """
    from qiskit_aer import AerSimulator
    from qiskit_algorithms.optimizers import SPSA
    
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_processed_data()
    vqc_template = build_vqc_circuit(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        feature_map_type=feature_map_type
    )
    sim = AerSimulator(method="statevector")
    
    x_params = sorted([p for p in vqc_template.parameters if p.name.startswith("x")], key=lambda p: p.name)
    theta_params = sorted(
        [p for p in vqc_template.parameters if p not in x_params],
        key=lambda p: int(p.name.split("[")[1].split("]")[0]) if "[" in p.name and p.name.split("[")[1].split("]")[0].isdigit() else p.name
    )
    
    history: List[Dict[str, Any]] = []
    np.random.seed(seed)
    
    def cost_function(theta: np.ndarray) -> float:
        # Stochastic subsample
        idx = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
        circs = []
        for x_val in X_train[idx]:
            b = {p: float(v) for p, v in zip(x_params, x_val)}
            b.update({p: float(v) for p, v in zip(theta_params, theta)})
            circs.append(vqc_template.assign_parameters(b))
            
        res = sim.run(circs, shots=shots).result()
        total_loss = 0.0
        for i, y_val in enumerate(y_train[idx]):
            counts = res.get_counts(i)
            tot = sum(counts.values())
            c1 = sum(cnt for k, cnt in counts.items() if k[-1] == "1")
            p1 = c1 / max(1, tot)
            total_loss += (p1 - float(y_val)) ** 2
            
        mse = total_loss / len(idx)
        history.append({
            "iteration": len(history),
            "loss": float(round(mse, 5)),
            "mean_energy": float(round(2.0 * mse - 1.0, 5))
        })
        return mse

    init_theta = np.random.uniform(-0.5, 0.5, len(theta_params))
    spsa = SPSA(maxiter=max_iter)
    res = spsa.minimize(cost_function, init_theta)
    
    optimal_theta = res.x
    np.save(os.path.join(output_dir, "optimal_theta.npy"), optimal_theta)
    
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    return optimal_theta, history


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = generate_and_save_data()
    print(f"Dataset generated: Train={X_tr.shape}, Test={X_te.shape}")
    vqc = build_vqc_circuit(num_qubits=4, reps=2, entanglement="sca")
    print(f"Upgraded VQC: Qubits={vqc.num_qubits}, Ops={dict(vqc.count_ops())}")
    print("Executing SPSA (150 iterations) training...")
    opt_theta, hist = train_vqc(max_iter=150)
    print(f"SPSA Optimization complete. Total evaluations logged: {len(hist)}, Final loss: {hist[-1]['loss']:.4f}")
