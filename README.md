# CONFIDE (ICSME 2026) — Anonymous Reproducibility Package

This repository is prepared for **double-anonymous review**.

## Terminology note (Semantic Refiner vs DADE)

In the paper, we refer to the semantic post-processing component as **Semantic Refiner (SR)**.
In the implementation, the same component is historically named **DADE** (e.g., in script names, variables, and some output filenames such as `*sem_dade*`).
Unless otherwise stated:

- **SR (paper)** ≡ **DADE (code)**

## What is included

- Source code for the CONFIDE pipeline (multi-modal similarity, **Semantic Refiner (SR)**, EDL uncertainty modeling, and conflict-aware clustering).
- A **curated snapshot** of the minimal inputs required to reproduce the paper’s main results (Table III/IV and main figures) under:
  - `data/processed/paper_inputs/<tag>/...`

## What is NOT redistributed

- Third-party subject systems (benchmark source code) and raw traces are **not** redistributed in this anonymous package.
  See `data/raw/README.md` for upstream download links.

## Quickstart (copy/paste commands)

> All commands below assume **Windows + PowerShell** and are executed at the repository root.

### 1) Create an environment and install dependencies

```powershell
cd d:\multimodal_microservice_extraction

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Use the provided paper input snapshot (recommended)

Pick a snapshot tag (default: `paper_v1`) and copy the snapshot back to the canonical `data/processed/...` locations expected by scripts:

```powershell
$env:CONFIDE_PAPER_TAG = "paper_v1"
$tag = $env:CONFIDE_PAPER_TAG

Copy-Item -Recurse -Force "data\processed\paper_inputs\$tag\data\processed\*" "data\processed\"
```

> If you already have the canonical `data/processed/...` files present, you can skip this copy.

### 3) Reproduce Table III (overall comparison)

```powershell
# Step 1: generate Phase3 partitions (CAC + baselines)
python scripts\multimodal\phase3\phase3_cac_evaluation.py acmeair daytrader jpetstore plants

# Step 2: run baselines + ours and generate the paper table
python scripts\multimodal\phase4\run_mono_baselines_and_ours_table.py daytrader plants acmeair jpetstore
python scripts\multimodal\phase4\make_all_systems_mono_table.py
python scripts\multimodal\phase4\generate_paper_final_table.py
```

Expected key outputs:
- `results/ablation/baseline/paper_final_table.csv`
- `results/ablation/baseline/paper_final_table.md`

### 4) Reproduce Table IV (diagnosis)

```powershell
python scripts\multimodal\phase4\generate_table_IV_regress_diagnosis.py
```

Expected outputs:
- `results/artifact_tables/table_IV_regress_diagnosis.csv`
- `results/artifact_tables/table_IV_regress_diagnosis.md`

### 5) Reproduce main figures

```powershell
# Semantic smoothing (bar)
python scripts\multimodal\phase4\plot_semantic_smoothing_bar_median_iqr.py

# Semantic PDF before/after Semantic Refiner (SR)
python scripts\multimodal\phase4\plot_semantic_pdf_dade.py
```

Expected (common) figure output folders:
- `results/paper/` (when a script is invoked with `--paper_mode`)
- `results/plots/` (default for some plotting scripts)

On Windows, the generated figures will be under:
- `D:\multimodal_microservice_extraction\results\paper`
- `D:\multimodal_microservice_extraction\results\plots`

### One-command reproduction (recommended)

After installing dependencies, you can reproduce **Table III/IV + main figures** with a single PowerShell command:

```powershell
.\scripts\reproduce_paper.ps1 -Tag paper_v1
```

This script will (1) copy the snapshot back to `data/processed/...`, then (2) run Phase3/Phase4/plotting steps, and finally (3) print whether the expected output files exist.

## Optional: regenerate a snapshot

Only needed if you regenerated canonical inputs locally and want to record hashes:

```powershell
python scripts\multimodal\phase4\snapshot_paper_inputs.py --tag audit_local
```

The snapshot will be written under `data/processed/paper_inputs/audit_local/`.

## Notes on paths

Most scripts assume the repository root as the working directory and use relative paths such as `data/processed/...`.

## Pipeline overview (dataset construction)

This package includes an automated pipeline that constructs high-fidelity multi-modal datasets from heterogeneous sources:

- **Structural modality (static structure)**
  - Java-based extractors live under `tools/` (built as a fat JAR via Maven).
  - `scripts/structural/` invokes these extractors to generate base artifacts such as AST, call graph, and dependency graph (written under `data/processed/{ast,callgraph,dependency}/`).

- **Semantic modality (code semantics)**
  - `scripts/semantic/` uses the same `tools/target/tools-fat.jar` to extract method-level semantic signals (e.g., identifiers/comments/variables), then builds embeddings and semantic similarity matrices.

- **Multi-modal similarity matrices (Phase 1 fusion inputs)**
  - `scripts/multimodal/phase1/` (notably `build_multimodal_matrices.py`) constructs the aligned similarity matrices for each modality (semantic/structural/temporal) and writes them to `data/processed/fusion/`.

- **Temporal modality (runtime behavior)**
  - Download and run the four monolith applications, start the stack under `docker/` (via `docker/docker-compose.yml`), and execute the JMeter workloads documented in `scripts/jmeter/readme`.
  - This produces JTL logs under `results/jmeter/` and trace exports; then `scripts/temporal/` builds the temporal similarity matrix (e.g., `scripts/temporal/build_S_temp.py` writes `data/processed/temporal/<system>_S_temp.npy`).

### Claimed contribution (artifact)

We develop an automated pipeline to construct high-fidelity multi-modal datasets from heterogeneous sources. By integrating non-intrusive instrumentation with domain-aware preprocessing, our pipeline provides auditable benchmarks that address the critical scarcity of multi-modal data and support reproducible research.
