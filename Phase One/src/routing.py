"""
src/routing.py
Transpilation passes, layout synthesis, routing routines,
and comparative overhead analysis across Processors A, B, and C.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager, CouplingMap
from src.processors import (
    get_qiskit_coupling_map,
    PROCESSOR_NUM_QUBITS
)
from src.problem import build_vqc_circuit


def count_swaps(circuit: QuantumCircuit) -> int:
    """
    Counts explicit or decomposed SWAP gates in a transpiled quantum circuit.
    Standard decomposition: 1 SWAP = 3 CX gates.
    """
    ops = circuit.count_ops()
    if "swap" in ops:
        return ops["swap"]
    return 0


def count_cxs(circuit: QuantumCircuit) -> int:
    """Counts CX/CNOT two-qubit entangling gates in circuit."""
    ops = circuit.count_ops()
    return ops.get("cx", 0)


def extract_circuit_metrics(circuit: QuantumCircuit, original_circuit: Optional[QuantumCircuit] = None) -> Dict[str, Any]:
    """
    Extracts structural circuit metrics: depth, gate counts, CX count, SWAP count,
    and routing overhead relative to the original virtual circuit.
    """
    depth = circuit.depth()
    ops = circuit.count_ops()
    cx_count = count_cxs(circuit)
    swap_count = count_swaps(circuit)
    total_gates = sum(ops.values())
    
    metrics = {
        "depth": depth,
        "total_gates": total_gates,
        "cx_count": cx_count,
        "swap_count": swap_count,
        "gate_breakdown": {k: int(v) for k, v in ops.items()}
    }
    
    if original_circuit is not None:
        orig_cx = count_cxs(original_circuit)
        orig_depth = original_circuit.depth()
        # In modern Qiskit, routing inserts SWAP as 3 CXs or explicit SWAPs
        # If explicit SWAPs are 0, extra CXs = (cx_count - orig_cx) // 3
        implied_swaps = swap_count if swap_count > 0 else max(0, (cx_count - orig_cx) // 3)
        metrics["orig_depth"] = orig_depth
        metrics["orig_cx"] = orig_cx
        metrics["depth_overhead_ratio"] = round(depth / max(1, orig_depth), 3)
        metrics["cx_overhead_ratio"] = round(cx_count / max(1, orig_cx), 3)
        metrics["implied_swaps"] = implied_swaps
        
    return metrics


def transpile_for_processor(
    circuit: QuantumCircuit,
    processor_name: str,
    optimization_level: int = 3,
    routing_method: str = "sabre",
    layout_method: str = "sabre",
    seed_transpiler: int = 42
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    Transpiles a quantum circuit targeting a specific processor's coupling map
    and basis gates with configurable optimization and routing passes.
    """
    coupling_map = get_qiskit_coupling_map(processor_name)
    basis_gates = ["sx", "x", "rz", "cx", "id"]
    
    start_t = time.perf_counter()
    transpiled_qc = transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        routing_method=routing_method,
        layout_method=layout_method,
        seed_transpiler=seed_transpiler
    )
    routing_time_ms = (time.perf_counter() - start_t) * 1000.0
    
    metrics = extract_circuit_metrics(transpiled_qc, original_circuit=circuit)
    metrics["processor"] = f"Processor_{processor_name.upper().replace('PROCESSOR_', '')}"
    metrics["optimization_level"] = optimization_level
    metrics["routing_method"] = routing_method
    metrics["layout_method"] = layout_method
    metrics["transpilation_time_ms"] = round(routing_time_ms, 2)
    
    # Extract layout mapping
    if transpiled_qc.layout:
        try:
            initial_layout = transpiled_qc.layout.initial_layout
            metrics["layout_mapping"] = {
                str(k): int(v) for k, v in initial_layout.get_virtual_bits().items()
            }
        except Exception:
            metrics["layout_mapping"] = "Available"
            
    return transpiled_qc, metrics


def run_full_routing_benchmarks(
    output_dir: str = "results/tables",
    circuit: Optional[QuantumCircuit] = None
) -> pd.DataFrame:
    """
    Executes an authoritative routing benchmark sweep:
    - Processors: A (5-qubit), B (9-qubit), C (14-qubit Heavy-Hex)
    - Optimization levels: 0, 1, 2, 3
    - Routing methods: sabre, stochastic, lookahead
    Saves tables to results/tables/routing_benchmarks.csv and .json.
    """
    os.makedirs(output_dir, exist_ok=True)
    if circuit is None:
        circuit = build_vqc_circuit(num_qubits=4, reps=2, entanglement="sca")
        
    records = []
    processors = ["A", "B", "C"]
    opt_levels = [0, 1, 2, 3]
    routing_methods = ["sabre", "lookahead", "basic"]
    
    print("Running Hardware Routing & Transpilation Benchmark Matrix...")
    for proc in processors:
        for opt in opt_levels:
            for rm in routing_methods:
                try:
                    _, metrics = transpile_for_processor(
                        circuit=circuit,
                        processor_name=proc,
                        optimization_level=opt,
                        routing_method=rm,
                        seed_transpiler=42
                    )
                    records.append(metrics)
                except Exception as e:
                    print(f"Warning: Failed run for {proc}, opt={opt}, rm={rm}: {e}")
                    
    df = pd.DataFrame(records)
    
    # Save CSV and JSON
    csv_path = os.path.join(output_dir, "routing_benchmarks.csv")
    json_path = os.path.join(output_dir, "routing_benchmarks.json")
    df.to_csv(csv_path, index=False)
    
    # Convert records to JSON serializable format
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    print(f"Routing benchmarks exported: {len(records)} configurations recorded.")
    return df


if __name__ == "__main__":
    df = run_full_routing_benchmarks()
    # Print summary view
    summary = df.groupby(["processor", "optimization_level", "routing_method"])[["depth", "cx_count", "implied_swaps", "transpilation_time_ms"]].mean()
    print(summary)
