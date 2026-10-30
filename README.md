# Qiskit Fall Fest 2026 — Track 4 (Quantum Machine Learning)
## Problem Statement 3 (PS-Q3): Quantum Circuit Learning on Noisy Quantum-Generated Data

### Project Overview & Phase II Declarations
- **Track:** Track 4 — Quantum Machine Learning (QML)
- **Problem Statement:** PS-Q3 (`QML-QCIRCUIT-NOISE` / `Q3-NOISE-BENCH`)
- **Common Task A:** Bell-State Entanglement Preparation & Correlation Fidelity (`|Φ⁺⟩ = (|00⟩ + |11⟩)/√2`)
- **Common Task B:** Processor D Architecture & QFT-3 Topological Benchmark (`processor_d_config.json`)
- **Noise Model:** Organizer Benchmark Noise Model **QFF2026-v2** (`noise_model/benchmark_noise_model.py`)
- **Enhanced Quantum Architecture:** Linear Angle Encoding ($\theta_i \propto x_i$) + EfficientSU2 ($reps=3$, 32 parameters) + Quasi-Newton L-BFGS-B Optimizer
- **Key Performance Achievement:** Quantum VQC reaches **0.9181 ROC-AUC** (Ideal) and **0.8992 ROC-AUC** (Noisy REM Mitigated), closing 97% of the performance gap with Classical Logistic Regression (0.9248 ROC-AUC) while reducing seed variance by **14.6x** ($0.9142 \pm 0.0078$).
- **Mitigation Bonus Engine:** Method 1: Readout Error Mitigation via Pre-flight Inversion & Simplex Projection (Prioritized Primary); Method 2: Zero-Noise Extrapolation via Mitiq Global Folding (Deprioritized due to depth-induced depolarizing gate noise accumulation).

---

### Workspace Directory Structure
The repository strictly isolates legacy Phase I artifacts and maintains the Phase II implementation directly in the root directory:

```
project/ (Repository Root)
├── Phase One/              # Preserved legacy Phase I notebooks, source code, and processors
├── benchmark/              # Official Task A and Task B evaluation suite (qff_qml_benchmark.py)
├── data/q3/                # True PS-Q3 challenge data (qnoise_train.csv, qnoise_val.csv, qnoise_test_features.csv)
├── noise_model/            # Official organizer noise model (benchmark_noise_model.py, QFF2026-v2)
├── schemas/                # Processor D JSON schema & Q3 feature dictionaries
├── src/
│   ├── __init__.py
│   ├── problem.py          # 24-feature dataset handler, non-linear angle encoding, & EfficientSU2 ansatz
│   └── metrics.py          # Dual Mitigation Engine (REM matrix inversion + Mitiq ZNE)
├── tools/                  # Official predictions validation tool (validate_predictions.py)
├── README.md               # Track 4/PS-Q3 declaration, execution directives, & pip freeze
├── solution.ipynb          # Primary executable notebook for the judging panel
├── processor_d_config.json # Validated Task B Processor D specifications & QFT-3 results
├── predictions_q3.csv      # Formatted test set predictions (sample_id, label, score)
└── results.md              # Official technical report matching results_template.md
```

---

### Processor D Specifications (Task B)
- **Processor ID:** `processor_d_q3_ring5_v1`
- **Qubits:** 5 physical superconducting qubits (Nodes: 0, 1, 2, 3, 4)
- **Coupling Map:** Bidirectional ring topology minimizing routing SWAP gate insertion:
  `[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 0], [0, 4]]`
- **Native Gate Set:** `['rx', 'ry', 'rz', 'h', 'x', 'sx', 'cx']` (All covered by benchmark noise model)
- **Design Rationale:** A 5-qubit bidirectional ring graph matches the 4-qubit ring-CNOT entanglement layout of the PS-Q3 classification circuit, completely eliminating routing SWAP gate insertion overhead while natively supporting the 3-qubit QFT benchmark structure.

---

### Reproduction & Execution Guide

#### 1. Self-Test & Task B Benchmark
Execute the official Task B benchmark to transpile and evaluate the 3-qubit QFT:
```bash
python benchmark/qff_qml_benchmark.py processor_d_config.json
```

#### 2. Validate Test Predictions File
Verify that `predictions_q3.csv` strictly adheres to organizer requirements (500 rows, `sample_id,label,score`, probabilities in [0, 1]):
```bash
python tools/validate_predictions.py q3 predictions_q3.csv
```

#### 3. Run Solution Notebook
Open and execute the top-to-bottom judging notebook:
```bash
# Interactive GUI
jupyter notebook solution.ipynb

# Or headless execution via nbconvert:
python -m nbconvert --to notebook --execute solution.ipynb --output solution.ipynb
```

---

### Pinned Environment (`pip freeze`)
Full environment snapshot saved to [`pip_freeze.txt`](file:///d:/QFF%20SRM%20AP/pip_freeze.txt):

```text
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
asttokens==3.0.2
attrs==26.1.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.4.9
cirq-core==1.7.0
click==8.4.2
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.3.3
cpip==0.9.9
cpsat-logutils==1.0.0
cryptography==50.0.0
cycler==0.12.1
db-dtypes==1.7.1
debugpy==1.8.21
decorator==5.3.1
dill==0.4.1
docplex==2.32.264
duet==0.2.9
dulwich==1.2.17
executing==2.2.1
fastapi==0.141.1
filelock==3.32.2
fonttools==4.63.0
fsspec==2026.7.0
google-auth==2.56.3
google-crc32c==1.8.0
google-resumable-media==2.10.1
googleapis-common-protos==1.75.1
greenlet==3.5.4
grpcio==1.83.0
grpcio-status==1.83.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
huepy==1.2.1
huggingface_hub==1.26.0
idna==3.18
ipykernel==7.3.0
ipython==9.15.0
ipython_pygments_lexers==1.1.1
jedi==0.20.0
Jinja2==3.1.6
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_client==8.9.1
jupyter_core==5.9.1
kiwisolver==1.5.0
livereload==2.7.1
llvmlite==0.48.0
lxml==6.1.3
MAPIE==1.5.0
markdown-it-py==4.2.0
markdown2==2.5.5
MarkupSafe==3.0.3
matplotlib==3.11.1
matplotlib-inline==0.2.2
mdurl==0.1.2
mitiq==0.25.0
mpmath==1.3.0
narwhals==2.24.0
nest-asyncio2==1.7.2
networkx==3.6.1
numba==0.66.0
numpy==2.4.6
orloge==1.0.0
packaging==26.2
pandas==3.0.5
parso==0.8.7
pillow==12.3.0
platformdirs==4.10.0
ply==3.11
prompt_toolkit==3.0.52
proto-plus==1.28.3
protobuf==7.35.1
psutil==7.2.2
PuLP==4.0.0
pure_eval==0.2.3
pyarrow==25.0.0
pyasn1==0.6.4
pyasn1_modules==0.4.2
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-dotenv==1.2.4
python-louvain==0.16
python-multipart==0.0.32
python-pptx==1.0.2
PyYAML==6.0.3
pyzmq==27.1.0
qiskit==2.5.2
qiskit-aer==0.17.2
qiskit-algorithms==0.4.0
qiskit-optimization==0.7.0
referencing==0.37.0
regex==2026.7.19
reportlab==5.0.1
requests==2.34.2
rich==15.0.0
rpds-py==2026.9.1
rustworkx==0.18.1
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.18.0
sentence-transformers==6.0.1
setuptools==83.0.0
shap==0.52.0
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
slicer==0.0.8
SQLAlchemy==2.0.51
stack-data==0.6.3
starlette==1.4.1
stevedore==5.9.1
sympy==1.14.0
threadpoolctl==3.6.0
tifffile==2026.9.9
tokenizers==0.22.2
torch==2.13.0+cpu
tornado==6.5.7
tqdm==4.70.0
traitlets==5.15.1
transformers==5.14.1
typer==0.27.1
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.3
urllib3==2.7.0
uvicorn==0.52.1
vite==1.5.2
watchfiles==1.3.0
wcwidth==0.8.2
websockets==17.0.1
xlsxwriter==3.2.9
```
