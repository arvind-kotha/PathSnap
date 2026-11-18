# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University | Department of Computer Science and Engineering**  
**Course:** UE24CS352A — Machine Learning Hackathon  
**Two-Laptop Architecture:** Laptop 1 (Model API Server) & Laptop 2 (Evaluator Console)

---

## 1. Project Overview

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a directed road network graph. The system integrates:
* **Spatial Indexing & Candidate Generation**: Local planar metric projection and $k$-d tree candidate search.
* **Feature Engineering**: 11 spatial, angular, and kinematic features per GPS fix / road segment pair.
* **Supervised Candidate Scoring**: 5 evaluated model families (Logistic Regression, Gaussian Naive Bayes, Random Forest, HistGradientBoosting, Shallow MLP).
* **Bayesian Hyperparameter Optimization**: 20-trial Optuna TPE search with group-aware cross-validation.
* **Sequential Route Reconstruction (HMM + Viterbi)**: Gap-aware Viterbi decoding with graph topological transitions ($p_{\text{stay}} = 0.60$, $p_{\text{direct}} = 0.38$, $p_{\text{jump}} = 10^{-5}$) and GPS dropout handling ($\Delta t > 30\,\text{s}$).
* **Distributed Deployment**: High-performance FastAPI backend on Laptop 1 with isolated server-side ground truth, and an interactive Leaflet web console on Laptop 2.

---

## 2. Repository Structure

```text
PathSnap/
├── artifacts/                         # Model weights and experimental evaluation logs
│   ├── pathsnap.joblib                # Active trained Random Forest model (11 features)
│   ├── verified_evaluation_results.json # Full verification metrics across student & evaluator partitions
│   └── api_test_results.json          # End-to-end API integration verification log
├── evaluator_ui/                      # Evaluator Web Console (served by FastAPI & runs on Laptop 2)
│   ├── index.html                     # Responsive evaluation dashboard
│   ├── style.css                      # Modern dark theme styles
│   └── app.js                         # Leaflet map logic & API client
├── notebooks/                         # Complete experimental research notebooks
│   └── PathSnap_Complete_Final_5_Models.ipynb # Training & tuning notebook for all 5 models
├── outputs/                           # Generated reports and geographical visualizations
│   ├── PathSnap_Final_Technical_Report.pdf # 8-page final consolidated technical report
│   ├── pathsnap_00000049.html         # Interactive Folium map for Evaluator Trip 00000049
│   ├── pathsnap_00000000.html         # Interactive Folium map for Development Trip 00000000
│   └── report_assets/                 # Figures (Fig 1–5) embedded in technical report
├── scripts/                           # Execution, evaluation, and report generation scripts
│   ├── build_technical_report_pdf.py  # Generates the official ReportLab PDF deliverable
│   ├── verify_all_partitions.py       # Benchmarks raw vs HMM on student and evaluator sets
│   ├── test_api_integration.py        # End-to-end FastAPI test client on all 10 evaluator trips
│   └── generate_report_assets.py      # Matplotlib figure generation
├── src/pathsnap/                      # Core production Python package
│   ├── __init__.py
│   ├── api.py                         # FastAPI service & static file server
│   ├── core.py                        # Spatial indexing, candidate scoring & Viterbi engine
│   ├── map.py                         # Interactive Folium map generation
│   └── train.py                       # CLI training pipeline
├── pyproject.toml                     # Python package metadata and CLI entry points
├── .gitignore                         # Strict exclusion for .venv and evaluation datasets
└── README.md                          # Project documentation and deployment guide
```

---

## 3. Installation & Setup

From PowerShell in the repository root:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install package in editable mode
python -m pip install --upgrade pip
python -m pip install -e .
```

---

## 4. Model Performance & HMM Verification Results

### 4.1 Five-Model Comparison (Development Partition)
All 5 models were trained across 80 training trips using 3-fold trip-grouped cross-validation:

| Model | Family | CV Top-1 | CV Macro-F1 | Student-Test Raw Top-1 | Student-Test Decoded Top-1 | Log Loss | Latency (ms/1k) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | Linear / Probabilistic | 64.88% | 71.33% | 78.22% | 77.30% | 0.3399 | 1.2 ms |
| **Gaussian Naive Bayes** | Generative | 65.67% | 79.32% | 77.61% | 74.38% | 0.3157 | 0.8 ms |
| **Random Forest (Active)** | Ensemble (Bagging) | 67.77% | 83.29% | 82.21% | **88.43%** | 0.1629 | 5.4 ms |
| **HistGradientBoosting** | Ensemble (Boosting) | 67.01% | 84.40% | 82.39% | 82.50% | 0.1438 | 3.1 ms |
| **Shallow MLP** | Neural Network | 66.64% | 83.85% | 82.29% | 82.38% | 0.1602 | 2.8 ms |

### 4.2 Step 7 HMM-Viterbi Verification (Evaluator Test Partition)
Evaluated across all 10 reserved evaluation trips (`00000015` – `00000094`) with ground truth verified strictly server-side:

| Trip ID | GPS Fixes | Raw Top-1 Acc | Decoded Top-1 Acc | Accuracy Gain | Decoded Macro-F1 | Decoded Route Continuity | Inference Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **00000015** | 6,854 | 54.36% | **73.69%** | **+19.33%** | 41.25% | 99.77% | 4,885 ms |
| **00000017** | 1,346 | 85.51% | **83.51%** | -2.01% | 66.78% | 95.68% | 1,387 ms |
| **00000023** | 458 | 90.61% | **93.45%** | **+2.84%** | 74.14% | 94.20% | 474 ms |
| **00000030** | 654 | 77.68% | **79.20%** | **+1.53%** | 59.19% | 96.11% | 788 ms |
| **00000049** | 2,642 | 96.03% | **98.56%** | **+2.54%** | 93.74% | 99.76% | 2,021 ms |
| **00000059** | 1,031 | 90.59% | **89.04%** | -1.55% | 70.57% | 96.33% | 871 ms |
| **00000067** | 916 | 67.14% | **76.97%** | **+9.83%** | 62.92% | 100.00% | 832 ms |
| **00000073** | 1,144 | 89.16% | **97.29%** | **+8.13%** | 90.69% | 99.19% | 1,042 ms |
| **00000087** | 1,399 | 91.78% | **97.71%** | **+5.93%** | 92.80% | 100.00% | 2,036 ms |
| **00000094** | 3,230 | 66.10% | **77.99%** | **+11.89%** | 67.80% | 99.54% | 2,716 ms |
| **MEAN** | **2,060** | **80.90%** | **86.74%** | **+5.85%** | **71.99%** | **98.06%** | **1,705 ms** |

---

## 5. Deployment Instructions (Two-Laptop Setup)

### Laptop 1 (Model & FastAPI Backend)
1. Ensure the evaluator data folder is at `data/evaluator_test` (or configure via environment variable).
2. Start the FastAPI server:
   ```powershell
   uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
   ```
3. Laptop 1 displays live request logs showing fix count, inference latency, and accuracy metrics.

### Laptop 2 (Evaluator Web Console)
1. Connect Laptop 2 to the same network as Laptop 1.
2. Open a web browser to:
   ```text
   http://<LAPTOP-1-IP>:8000
   ```
   *(Or unzip `artifacts/PathSnap_Laptop2_UI.zip`, open `index.html`, and set the Laptop 1 IP in the top configuration bar).*
3. Select any of the 10 evaluation trips and click **"Run Map-Matching"** to observe live trajectory snapping and KPI metrics.

---

## 6. Deliverables & Documentation

* **Technical Report (PDF)**: Located at [`outputs/PathSnap_Final_Technical_Report.pdf`](outputs/PathSnap_Final_Technical_Report.pdf).
* **Interactive Folium Map**: Located at [`outputs/pathsnap_00000049.html`](outputs/pathsnap_00000049.html).
* **Evaluation Manifest**: Located at [`artifacts/final_export/manifest.json`](artifacts/final_export/manifest.json).
