"""
src/processors.py
Authoritative Processor A and Processor B topology definitions,
NetworkX graph builders, topological analysis routines, and unified
Qiskit Aer synthetic noise models for Qiskit Fall Fest 2026 Phase 1.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import json
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------------------------------------------------------
# PHASE 1 AUTHORITATIVE COUPLING MAP DEFINITIONS
# -------------------------------------------------------------------------
# Processor A (5 Qubits, Nodes 0 to 4): Branching tree topology
PROCESSOR_A_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (1, 3),
    (3, 4)
]

# Processor B (9 Qubits, Nodes 0 to 8): Sparse heavy-hex-inspired 12-edge mesh
PROCESSOR_B_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (1, 6),
    (2, 3),
    (2, 6),
    (2, 5),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (6, 8),
    (8, 0)
]

# Processor C (14 Qubits, Nodes 0 to 13): Heavy-hex Phase 2 proposal architecture
PROCESSOR_C_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    (4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 4),
    (2, 11), (11, 12), (12, 13), (13, 8)
]

PROCESSOR_NUM_QUBITS: Dict[str, int] = {
    "A": 5,
    "B": 9,
    "C": 14,
    "Processor_A": 5,
    "Processor_B": 9,
    "Processor_C": 14
}

AUTHORITATIVE_MAPS: Dict[str, List[Tuple[int, int]]] = {
    "A": PROCESSOR_A_EDGES,
    "B": PROCESSOR_B_EDGES,
    "C": PROCESSOR_C_EDGES,
    "Processor_A": PROCESSOR_A_EDGES,
    "Processor_B": PROCESSOR_B_EDGES,
    "Processor_C": PROCESSOR_C_EDGES
}


# -------------------------------------------------------------------------
# GRAPH CONVERSION & TOPOLOGICAL METRICS
# -------------------------------------------------------------------------
def get_bidirectional_edges(edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Converts a directed or undirected edge list into symmetric bidirectional pairs."""
    pair_set = set()
    for u, v in edges:
        pair_set.add((u, v))
        pair_set.add((v, u))
    return sorted([[u, v] for u, v in pair_set])


def build_networkx_graph(processor_name: str) -> nx.Graph:
    """Constructs a NetworkX undirected graph for Processor A or B."""
    key = processor_name.upper().replace("PROCESSOR_", "")
    if key not in ["A", "B", "C"]:
        raise ValueError(f"Unknown processor: {processor_name}. Supported: 'A', 'B', 'C'.")
    
    edges = AUTHORITATIVE_MAPS[key]
    num_qubits = PROCESSOR_NUM_QUBITS[key]
    
    G = nx.Graph()
    G.add_nodes_from(range(num_qubits))
    G.add_edges_from(edges)
    return G


def compute_topological_metrics(processor_name: str) -> Dict[str, Any]:
    """
    Computes rigorous graph-theoretic metrics for hardware architectural comparison:
    - Diameter, Radius, Average Shortest Path Length
    - Average Degree, Max Degree, Degree Distribution
    - Algebraic Connectivity (Fiedler eigenvalue of Laplacian matrix)
    - Node Connectivity & Edge Connectivity
    """
    G = build_networkx_graph(processor_name)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))
    max_degree = int(np.max(degrees))
    
    is_connected = nx.is_connected(G)
    if is_connected:
        diameter = nx.diameter(G)
        radius = nx.radius(G)
        avg_path_length = nx.average_shortest_path_length(G)
        center_nodes = nx.center(G)
        periphery_nodes = nx.periphery(G)
        algebraic_connectivity = float(nx.algebraic_connectivity(G, method="lanczos"))
        node_connectivity = int(nx.node_connectivity(G))
        edge_connectivity = int(nx.edge_connectivity(G))
    else:
        diameter = None
        radius = None
        avg_path_length = None
        center_nodes = []
        periphery_nodes = []
        algebraic_connectivity = 0.0
        node_connectivity = 0
        edge_connectivity = 0

    clustering = float(nx.average_clustering(G))
    
    return {
        "processor": f"Processor_{processor_name.upper().replace('PROCESSOR_', '')}",
        "num_qubits": num_nodes,
        "num_edges": num_edges,
        "is_connected": is_connected,
        "diameter": diameter,
        "radius": radius,
        "avg_path_length": round(avg_path_length, 4) if avg_path_length else None,
        "avg_degree": round(avg_degree, 3),
        "max_degree": max_degree,
        "clustering_coefficient": round(clustering, 4),
        "algebraic_connectivity": round(algebraic_connectivity, 4),
        "node_connectivity": node_connectivity,
        "edge_connectivity": edge_connectivity,
        "center_nodes": center_nodes,
        "periphery_nodes": periphery_nodes,
        "degree_dict": dict(G.degree())
    }


def get_qiskit_coupling_map(processor_name: str) -> CouplingMap:
    """Returns a Qiskit CouplingMap instance for the selected processor."""
    key = processor_name.upper().replace("PROCESSOR_", "")
    edges = AUTHORITATIVE_MAPS[key]
    bidi_edges = get_bidirectional_edges(edges)
    return CouplingMap(couplinglist=bidi_edges)


# -------------------------------------------------------------------------
# SYNTHETIC NOISE MODEL (0.1% 1q, 0.5% CX, 1.0% Readout Error)
# -------------------------------------------------------------------------
def get_unified_noise_model(
    gate_error_1q: float = 0.001,
    gate_error_2q: float = 0.005,
    readout_error_rate: float = 0.01
) -> NoiseModel:
    """
    Constructs the official Phase 1 Synthetic Noise Model:
    - 0.1% (0.001) single-qubit depolarizing error across rz, sx, x
    - 0.5% (0.005) two-qubit depolarizing error on cx
    - 1.0% (0.010) symmetric readout bit-flip error (P(1|0)=0.01, P(0|1)=0.01)
    """
    noise_model = NoiseModel()
    
    # 1-qubit depolarizing noise on native basis gates
    error_1q = depolarizing_error(gate_error_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["sx", "x", "rz"])
    
    # 2-qubit CX depolarizing noise
    error_2q = depolarizing_error(gate_error_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    
    # Symmetric readout error
    p1_0 = readout_error_rate
    p0_1 = readout_error_rate
    readout_err = ReadoutError([[1.0 - p1_0, p1_0], [p0_1, 1.0 - p0_1]])
    noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model


def get_aer_backend(
    processor_name: Optional[str] = None,
    noisy: bool = True,
    seed_simulator: int = 42
) -> AerSimulator:
    """
    Instantiates an AerSimulator configured with the processor's coupling map
    and the official synthetic noise model.
    """
    kwargs = {"seed_simulator": seed_simulator}
    
    if noisy:
        kwargs["noise_model"] = get_unified_noise_model()
    
    if processor_name is not None:
        cmap = get_qiskit_coupling_map(processor_name)
        kwargs["coupling_map"] = cmap
        kwargs["basis_gates"] = ["rz", "sx", "x", "cx"]
        
    return AerSimulator(**kwargs)


# -------------------------------------------------------------------------
# JSON EXPORT & SYNCHRONIZATION
# -------------------------------------------------------------------------
def export_processor_specs(output_dir: str = "processors") -> None:
    """Exports or refreshes JSON files for Processor A, Processor B, and Processor C."""
    os.makedirs(output_dir, exist_ok=True)
    for p in ["A", "B", "C"]:
        path = os.path.join(output_dir, f"processor_{p}.json")
        metrics = compute_topological_metrics(p)
        edges = AUTHORITATIVE_MAPS[p]
        bidi = get_bidirectional_edges(edges)
        data = {
            "name": f"Processor_{p}",
            "num_qubits": PROCESSOR_NUM_QUBITS[p],
            "basis_gates": ["rz", "sx", "x", "cx"],
            "authoritative_coupling_map": edges,
            "bidirectional_coupling_map": bidi,
            "synthetic_noise_model": {
                "gate_error_1q": 0.001,
                "gate_error_2q_cx": 0.005,
                "readout_error": 0.01
            },
            "graph_properties": metrics,
            "transpilation_baseline": {
                "optimization_level": 1,
                "seed": 42,
                "shots": 4096
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    export_processor_specs()
    print("Processors A, B, and C exported successfully.")
    for p in ["A", "B", "C"]:
        m = compute_topological_metrics(p)
        print(f"[{p}] Qubits: {m['num_qubits']}, Edges: {m['num_edges']}, Diameter: {m['diameter']}, AvgDegree: {m['avg_degree']}, AlgebraicConn: {m['algebraic_connectivity']}")
