# QualityTwin-T8377Z

**A quality-aware digital twin for production lines.**
Neurax Hackathon 3.0 · Domain 2: AI in Industry and Automation · Problem: *Visual Inspection & Defect Root-Cause Assistant*

QualityTwin is a software-only decision-support system. It connects three things that are normally analysed separately: **what is wrong with a product**, **where the production flow is constrained**, and **what that costs in throughput and profit**. Every output carries evidence, a calibrated confidence, and an *ADVISORY / SIMULATED* tag. Nothing touches hardware.

---

## 1. Problem Understanding

### The situation
A high-throughput, multi-stage line makes mixed product variants under changing inspection conditions. Defect signatures are subtle and easy to miss at production speed. Meanwhile, output is limited by cycle-time imbalance, excess work-in-process, downtime, changeovers, low utilization, scrap and rework. Quality, capacity and economics interact: a defect decision changes scrap and rework load, which changes flow, which changes margin. Models that look at only one of these cannot tell an operator what to change.

### What the system must answer
| # | Question | Our output |
|---|---|---|
| Q1 | Is this unit acceptable or defective, and which defect family? | Verdict + family |
| Q2 | Where is the defect? | Box / mask / heatmap, only where the data supports it |
| Q3 | How sure are we, and is this something never seen before? | Calibrated confidence; **UNCERTAIN / NOVEL** flag instead of a forced guess |
| Q4 | What process condition is likely responsible? | Evidence cards linking defect patterns to batch/process data |
| Q5 | Where is the flow constrained? | Bottleneck station with triangulated evidence |
| Q6 | What does it cost in throughput and money? | Loss waterfall, cost of quality, margin estimate |
| Q7 | What should change, and what happens to profit? | Ranked recommendations with expected impact and intervals |

### What we deliberately do not build
Not another image classifier, not a labeling tool, not an isolated KPI dashboard. No live camera feed, PLC connection, robotic sorting or machine control. All recommendations, bottleneck interventions and profit estimates are simulated or advisory.

### Our reading of the hard parts
- **"Flag uncertain or novel defect types instead of forcing a guess"** → open-set detection with abstention, evaluated by hiding a defect family during training.
- **"Localize defects where the data supports it"** → we pick the localization method by the label type available and state it (box, mask or heatmap). We never invent boxes.
- **Mixed variants, changing conditions, batch drift** → validation is grouped by batch, variant and lighting condition, so measured performance reflects unseen conditions.
- **"Continuously updated view"** → a streaming replay of time-ordered events in the dashboard.
- **"Unified"** → every screen links to the others: defect → batch → cause → station → dollars.

### Scope and assumptions
1. The provided production data is a **simulation** (Arena discrete-event model); see [Data](#4-data). It supports the flow, bottleneck and what-if layers directly.
2. Economic parameters that the organizers do not provide are stated as explicit assumptions, listed in the app, and can be edited.
3. Root-cause findings are **associations** with stated confidence, not proven causation.
4. The organizer-provided inspection images (12,000 grayscale PNGs in five class folders) carry no batch, variant, lighting or mask fields. Batch identifiers, process parameters and the defect↔process link are **simulated** by us and labeled as such everywhere in the app.

---

## 2. Architecture

```
 raw data ─► [Ingest + splits + dedup] ─► parquet tables
                    │
   ┌────────────────┼──────────────────────────────┐
   ▼                ▼                               ▼
 [Vision]        [Root cause]                 [Flow & bottleneck]
 detect/segment   batch table, drift,           station metrics, visit-adjusted
 anomaly maps     stratified tests,             utilization, queue/WIP, loss
 open-set         GBM + SHAP,                   waterfall, ML surrogate of the
   │              evidence cards                simulation for instant what-ifs
   ▼                │                               │
 [Calibration +     │                               ▼
  cost-based        │                        [Profit model + what-if grid]
  decision]         │                               │
   │                └───────────────┬───────────────┘
   └───────────────────────────────►▼
                     [Recommendations + evidence cards]
                                    ▼
                    artifacts/  (versioned parquet / json / png)
                                    ▼
                    [Streamlit dashboard, streaming replay]
```

**Design principles**
- **Batch pipeline → artifacts → thin UI.** Heavy compute writes versioned artifacts; the dashboard only reads them. It needs no GPU, no internet and no live model at demo time.
- **Contract-first.** The artifact schema is fixed early so modules can be built in parallel.
- **Evidence on every output.** Each verdict, cause, bottleneck and recommendation carries its evidence, confidence and an ADVISORY tag.
- **Reproducible.** Pinned dependencies, seeds, config-driven, one command per pipeline stage.

### Modules
| Module | Responsibility | Main outputs |
|---|---|---|
| `src/ingest` | Inventory, load, join, group-aware splits, duplicate detection | tables, `DATA_INVENTORY.md` |
| `src/vision` | Defect detection/classification, localization, anomaly maps, open-set | per-unit predictions, overlays |
| `src/calib` | Calibration, conformal abstention, cost-based accept/reject/review | calibrated probabilities, operating point |
| `src/rootcause` | Drift detection, stratified association tests, GBM + SHAP, evidence cards | `evidence_cards.json` |
| `src/flow` | Visit-adjusted utilization, bottleneck triangulation, loss waterfall, simulation surrogate | `flow_metrics.json`, scenario grid |
| `src/profit` | Margin model, intervals, what-if | `profit_model.json` |
| `src/recs` | Rank interventions by expected margin gain × evidence confidence | `recommendations.json` |
| `app/` | Streamlit dashboard | linked pages, live replay |

### Artifact contract (summary)
`predictions.parquet` · `batch_metrics.parquet` · `evidence_cards.json` · `flow_metrics.json` · `sim_results.parquet` · `profit_model.json` · `calibration.json` · `robustness.csv` · `recommendations.json`

### Dashboard pages
Line Health (live replay) · Inspect · Root Cause · Flow & Bottleneck · Profit & What-if · Trust (calibration, robustness, limitations, assumptions)

### Repository layout
```
src/{ingest,vision,calib,rootcause,flow,profit,recs}/   app/   artifacts/   notebooks/   tests/
config.yaml   .env.example   requirements.txt   README.md   DATA_INVENTORY.md
```

---

## 3. Approach

### 3.1 Inspection (vision)
- **Method by label type.** Boxes → detector. Masks → segmentation model. Image-level labels only → fine-tuned classifier with Grad-CAM and anomaly maps, reported as heatmap-based localization. Mostly-OK images → anomaly detection first, then defect-family classification.
- **On the provided images.** Labels are image-level (the class folder), so localization is heatmap-based and reported as such; we do not invent boxes. Images are 256 × 256, so we keep the native resolution and do not tile. The four defect families (crack, hole, rust, scratch) allow the novelty experiment to hide each family in turn.
- **Hybrid model.** (A) a supervised model for known defect families; (B) a patch-feature anomaly detector built from OK images (kNN memory bank on frozen pretrained features) for pixel-level anomaly maps; (C) an open-set score from distance to per-family prototypes. A small fusion model, trained only on out-of-fold predictions, outputs `OK`, `DEFECT(family)`, `UNCERTAIN` or `NOVEL`.
- **Small defects.** Input resolution and tiling are chosen from the measured defect-size distribution, so subtle defects are not downscaled away.
- **Robustness.** Augmentation modeled on the shifts named in the brief (lighting, orientation, batch and variant changes) plus test-time augmentation. A quick test checks whether images carry a batch signature the model could exploit instead of the defect.

### 3.2 Calibration and false-reject / false-accept handling
- Temperature scaling on a held-out *group* fold, then **split conformal prediction** to produce prediction sets. If the set is empty or ambiguous between OK and a defect class, the unit is flagged `UNCERTAIN` rather than guessed.
- **Cost-based three-way decision** using calibrated *p* = P(defective): accept costs *p*·c_escape, reject costs (1−*p*)·c_false_reject, review costs c_review; choose the minimum. Without a review option: reject when *p* > c_FR / (c_FR + c_escape). Costs come from the economic data where given, and are flagged as assumptions where not. The operating point is adjustable in the dashboard.

### 3.3 Flow and bottleneck analysis
For station *s* with *m* machines, cycle time CT, availability *A*, changeover loss fraction CO and visits per good unit *V* (which accounts for upstream scrap and rework loops):
- Maximum sustainable good throughput = min over *s* of `m·A·(1−CO) / (V·CT)`
- Utilization at demand λ: `ρ = λ·V·CT / (m·A·(1−CO))`
- **Triangulated call.** A station is named the bottleneck only when highest utilization, upstream queue/WIP build-up and downstream starvation agree. We report all three lines of evidence and recompute per product variant, since the constraint can shift with mix.
- **Loss waterfall.** Theoretical capacity → bottleneck capacity → downtime → changeover → scrap → rework → good units/hour, with OEE = availability × performance × quality. The data directly measures capacity and utilization/queue; downtime, changeover, scrap and rework rates are not present as fields and are entered as explicit, editable assumptions.
- **Cost of the constraint** follows Theory of Constraints: an hour lost at the bottleneck is an hour lost for the whole line, valued at throughput margin — margin itself is an assumed parameter, since no price/cost/profit field is provided.
- **Using the provided simulation data.** Models 1–2 sweep a `Demand` parameter with resulting KPIs per run (3,000 replicates each), which is what our ML surrogate (parameters → utilization, throughput, waiting time) trains on; it lets us check bottleneck calls against known simulation settings and power instant what-if sliders. Model 3 is a larger, ~20-station dataset used directly for station-level utilization/queue evidence.

### 3.4 Root-cause analysis
Two layers:
- **Flow root cause** (available from the provided data): sensitivity of utilization, queue time and throughput to the simulation's parameters.
- **Quality root cause** (needs defect labels linked to batch/process data): drift windows from control charts and change-point detection on process parameters; stratified association tests within product variant with Benjamini–Hochberg FDR correction; gradient-boosted models with SHAP and partial dependence to extract threshold statements; defect location/size patterns per batch to point at tooling.
- **Evidence card** for every hypothesis: variables, effect size, adjusted *p*, cross-validated lift, lag, stability across variants and time halves, confidence tier, the caveat "association, not proven causation", and how to verify. Negative results are reported too.

### 3.5 Profit and what-if
An accounting model from the economic inputs (margin per hour = good units × unit margin − scrap, rework, escape and downtime costs), cross-checked by an ML model trained on past periods and tested on later ones, with prediction intervals. Scenario outputs (more capacity at the bottleneck, less changeover, lower defect rate for one family, different inspection threshold) are pushed through the chosen model and shown with intervals.

### 3.6 Recommendations
Candidate actions (process setpoint corrections, capacity and flow moves, inspection-policy changes, mix and scheduling changes) are ranked by expected margin gain × evidence confidence. Each shows why, expected impact with interval, what could invalidate it, and how to verify. An optional language model may *narrate* the structured results, citing evidence ids; it never computes numbers, and a template fallback runs when it is unavailable.

### 3.7 Validation protocol
1. **Group-aware, frozen splits.** The images carry no batch, variant or lighting fields, so groups are built from the images: near-identical images (cosine ≥ 0.995 on 32 × 32 thumbnails) share a group. Splits are frozen in `artifacts/splits.csv` (`scripts/make_splits.py`, fixed seeds): a **condition holdout** (the darkest and brightest 7.5% of each class, about 15% of images) never used for training or model selection; a **locked holdout** (about 12%, stratified by class, grouped) evaluated at most three times; and a **dev** set (about 73%) with five stratified, grouped folds. No random splits.
2. **Evaluation discipline.** The locked holdout is touched only for final selection; the condition holdout is reserved for the unseen-lighting evaluation.
3. **Stress-test suite** (brightness, gamma, contrast, shadow gradients, blur, noise, JPEG, rotations, flips, scale) with results published in the Trust page.
4. **Novelty experiment:** hide one defect family (crack, hole, rust or scratch) in turn during training and show it is flagged `NOVEL`/`UNCERTAIN`.
5. **Metrics.** Classification: per-class precision/recall/F1, AUPRC, false-reject and false-accept rates, expected cost. Localization: no localization ground truth is provided, so heatmaps are assessed qualitatively and with stated proxies rather than IoU. Calibration: ECE, Brier, risk–coverage.
6. **Root-cause checks:** shuffled-label baseline and first-half/second-half stability.
7. **Flow checks:** surrogate and bottleneck calls compared with the simulation's known settings before any what-if is shown.

### 3.8 How this maps to the evaluation criteria
| Criterion | How we address it |
|---|---|
| Defect detection & classification accuracy | Hybrid model, group-aware validation, cost-aware operating point |
| Defect localization quality | Method matched to label type; IoU / Dice / PRO reported |
| Robustness to unseen conditions | Stress suite, shift-matched augmentation, leave-variant-out and novelty experiments |
| False-reject / false-accept handling | Three-way expected-cost decision, adjustable operating point |
| Root-cause correlation quality | Stratified, FDR-corrected, drift-aligned evidence cards |
| Explainability & confidence | Calibration, conformal abstention, evidence on every output, limitations page |
| Technical implementation | Batch pipeline → artifacts, pinned deps, one command per stage, clean-environment test |
| UI/UX & visualization | Linked pages, live replay, bottleneck evidence, what-if sliders |

### 3.9 Status
| Checkpoint | Status |
|---|---|
| CP1 | Problem analysis, architecture and approach (this document); repository, environment and automated data inventory in place |
| CP2 | In progress: ingest, flow v0 (station utilization and queue summaries), image inventory and triage, and the frozen split protocol are done; baseline vision training, the root-cause and profit layers, and the dashboard skeleton are next |
| CP3 | Planned: full evaluation, robustness table, calibration, dashboard, reproducible pipeline |

---

## 4. Data

**Provided production data:** *Manufacturing Data Shared Facility – Discrete-Event Simulation* (Mendeley Data, v2; generated with Arena Simulation; University of New South Wales; licence CC BY 4.0). Per our automated inventory, it ships three independent Arena experiments as CSVs, the underlying `.doe` models, process-model PDFs, and — for Model 3 — a fixed `ParametersFile.xls` of configured settings:

| File | Shape | One row is |
|---|---|---|
| `Model_1.csv` | 3,000 × 10 | One replicate at a sampled `Demand` (1–20), with resulting utilization, queue/wait time and throughput across 3 stations (Drilling, Milling, Assembly) |
| `Model_2.csv` | 3,000 × 17 | Same design, two part types, finer timing (VA/queue/storage per part, per station) |
| `Model_3.csv` | 605,620 × 78 | A larger, ~20-station dataset (Blanking, 4× Cell, Forklift, 2× Paint, 4× Press, Quality, Warehouse) with per-SKU VA/NVA/transport/wait time and per-station utilization/queue; `Time_Now` is 24 in every row, so each row is an independent replicate recorded at one simulated time, not a time series |

Models 1–2 sweep a `Demand` parameter with resulting KPIs per row — this is what supports the ML surrogate in §3.3. Model 3 has no swept per-row parameter and no time variation; it is used as a station-level source for bottleneck evidence (distributions and correlations across replicates), not for surrogate training. `Model_1.csv` carries 12 empty trailing columns (trailing commas), which the loader drops. The full column-level inventory is in [`DATA_INVENTORY.md`](DATA_INVENTORY.md), generated by `src/ingest/inventory.py`.

**What the data does not contain.** No cost, price, profit, revenue, scrap, rework, defect, downtime, changeover, breakdown, batch, lot, or timestamp field, in any file. A `Quality` station's utilization/queue is recorded, but not a pass/fail or defect-rate value. Economic parameters, batch identifiers, and any defect↔process link are therefore explicit assumptions we introduce — the flow and bottleneck layers are grounded in real simulation data; the quality and profit coupling on top of them is simulated.

**Inspection data (organizer-provided).** 12,000 PNG images, all 256 × 256 single-channel grayscale, in five class folders: `crack`, `hole`, `rust`, `scratch` and `normal`, exactly 2,400 images each. Labels are image-level (the folder name). There are no boxes, masks or per-image metadata such as batch, variant, lighting or timestamp. File names are `<class>_<number>.png`; the numbers have gaps and do not track brightness or any drift. Our triage (`artifacts/IMAGE_TRIAGE.md`) found:
- No exact duplicates. Near-identical images are rare among defect images; 21 small groups (largest: 8 images) are kept together in every split.
- Brightness and background vary widely inside every class, and brightness alone does not predict the class (five-class accuracy 0.22 against 0.20 by chance; normal-vs-defect AUROC 0.53). Five global statistics (mean, spread, extremes, edge energy) reach 0.81 five-class accuracy and 0.96 normal-vs-defect AUROC, because they measure the defects themselves. The task is easy in aggregate; the hard cases are thin cracks and faint marks.
- Smooth backgrounds and uniform defect styles look consistent with synthetic rendering. The organizers do not state the provenance, so our results describe method behavior on this data, not on a specific factory.
- The set is balanced by construction (80% defective), unlike a production line. Decision thresholds are reported at the dataset's prevalence and at an assumed, editable, realistic defect rate.

**Simulated coupling layer.** To connect inspection, flow and economics we generate the following and label all of it SIMULATED: batch identifiers and timestamps, process parameters with drift, a defect probability that depends on those parameters, and the economic parameters (`config.yaml`, editable in the app). Images are drawn into batches at the simulated defect rates, so the vision model's predictions are real while the defect↔process link is planted by construction. The root-cause module is therefore validated by recovering the planted effects against permutation baselines. We do not present its output as a discovery about a real factory.

**Handling.** Raw data is treated as read-only and is not committed to the repository. Nothing in the raw files was modified; all derived tables and features are produced by the pipeline under `artifacts/`.

---

## 5. Setup

Requirements: Windows 10/11 with PowerShell, Python 3.10 or newer. Data preparation and the flow layer run on CPU; vision training uses a cloud GPU.

1. Create the virtual environment:

```powershell
python -m venv .venv
```

2. Install dependencies:

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

3. Download the simulation dataset from the DOI in [References](#7-references) into `data/Model 1`, `data/Model 2` and `data/Model 3` (git-ignored), and place the organizer-provided inspection images in `data/images/train/<class>/`.

4. Generate the data inventory:

```powershell
.venv\Scripts\python.exe src/ingest/inventory.py
```

Image checks and the frozen split (each takes the image folder path): `scripts/image_inventory.py`, `scripts/image_triage.py`, `scripts/make_splits.py`. The committed `artifacts/splits.csv` is the protocol; do not regenerate it with different seeds.

5. Optional: create a local environment file for optional integrations (for example, the language-model narrator). The pipeline itself does not require any key.

```powershell
Copy-Item .env.example .env
```

Configuration lives in `config.yaml`. `.env` is never committed; only `.env.example` (placeholders) is.

### Reproduce in one command

From a checkout with the offline data/artifacts available, the complete artifact
path is:

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\python.exe run_pipeline.py --stage all
streamlit run app.py
```

`--stage all` runs the idempotent stages in dependency order, writes one log per
stage under `artifacts/logs/`, writes `artifacts/status.json`, and finishes with
contract validation. If a Kaggle output folder is present, select it with
`.venv\Scripts\python.exe run_pipeline.py --stage all --kaggle-run <run-folder>`;
otherwise existing artifacts are used when they are current. Use
`--force` to rebuild a stage. The normal infer path is explicitly offline and
CPU-only:

```powershell
.venv\Scripts\python.exe run_pipeline.py --stage infer --input data\images\train --output artifacts
```

It requires shipped model weights and calibration artifacts; it does not tune
thresholds or temperature. The locked holdout is never part of `--stage all`.
It has a separate human-gated evaluator and must not be used for tuning:
`.venv\Scripts\python.exe scripts\eval_locked_holdout.py --confirm`.

**Current build note.** Root Cause, Profit & What-if, Recommended Actions, and
Trust render from the existing evidence, profit, recommendation, calibration,
status, and final holdout artifacts; none of the four pages is soft-failed. The
complete `--stage all` command currently stops at contract validation because
`kaggle_downloads` has no OOF prediction inputs to rebuild calibration/novelty,
and several existing legacy artifacts do not satisfy the current schemas. The
Trust page shows that validation state and only displays values present in the
artifacts. The locked holdout result is final and was not rerun or used for
tuning.

### Assumptions register

The organizer files contain image-level labels and production simulation data,
but no joined batch/process/economic fields. Therefore any cross-layer result
is labeled `ADVISORY / SIMULATED` and says “association, not proven causation”.

| ID | Register entry |
|---|---|
| D14 | Rework coupling uses `rho_new = rho_baseline × (1 + rework_rate)`; rework is not observed in the production files. |
| D18 | Quality-to-flow and economics coupling is simulated/advisory; economics are read from `config.yaml`. |
| D22 | Price, unit, scrap, rework, escape, review and downtime costs are explicit `ASSUMED` values because no economics table was provided. |
| D24 | Simulated batches/process parameters are used to test root-cause recovery; the planted link is not a factory discovery. |
| D25 | Simulated timestamps, variants and station mappings are labeled assumptions and are not treated as organizer metadata. |

---

## 6. Limitations

- The provided production data is **synthetic** (Arena discrete-event simulation). It contains no real sensor noise or drift, so it supports the flow, bottleneck and what-if layers rather than defect visuals.
- Economic parameters not supplied by the organizers are assumptions, listed in the app; profit figures are estimates with intervals, not forecasts.
- No batch, defect, scrap/rework, downtime, changeover or cost field exists in the production data; the vision-to-flow coupling is a modeled assumption, not a joined field.
- Root-cause results are associations. Confounding by variant, time or shift is controlled by stratification but cannot be fully excluded.
- The simulation surrogate is only as valid as the simulation it imitates; we report its out-of-sample error and do not extrapolate beyond the parameter ranges in the data.
- The inspection images have image-level labels only, no metadata, and look synthetic. Vision results describe method behavior on this data; we report them on the locked and condition holdouts, not on the training distribution.
- The dataset is class-balanced (80% defective); real lines have far fewer defects, so decision thresholds are also reported at an assumed defect rate.
- Localization is heatmap-based with no ground truth to score it against.
- The defect↔process link and batch structure are simulated; root-cause results show that the method recovers planted effects, not real process causes.
- All recommendations are advisory and simulated. The system has no connection to machines, cameras or PLCs.

---

## 7. References

- Rabaev, M., Pratama, H., & Chan, K. C. (2019). *Manufacturing Data Shared Facility – Discrete-Event Simulation* [Data set]. Mendeley Data, V2. https://doi.org/10.17632/3rw227zxt7.2. Licensed under CC BY 4.0. The raw data was not modified.
- Chan, K. C., Rabaev, M., & Pratama, H. (2022). Generation of synthetic manufacturing datasets for machine learning using discrete-event simulation. *Production & Manufacturing Research*, 10(1), 337–353. https://doi.org/10.1080/21693277.2022.2086642
