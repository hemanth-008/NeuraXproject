# NEURAX 3.0 — Domain 2 · Visual Inspection & Defect Root-Cause Assistant
## MASTER PLAN (24 h) — paste this into any AI chat as ground truth

Living document. Sections 0 (Current status), 5A (What the data actually is), 11 (Mistakes Ledger), 12 (Decisions Log) and 14 (Progress Tracker) are updated as we go. Everything else is the strategy.

**Working rules / environment:** Windows + PowerShell. Paste terminal commands ONE AT A TIME (pasted multi-line blocks have merged into one line before). Keep the repo and the data OUTSIDE OneDrive (actual location: `C:\NeuraxHack`) — OneDrive syncing `.git` plus thousands of images causes lock errors and slowdowns; git remote is the backup. The desktop GPU (GT 610) cannot train anything: it is an editor + Streamlit host. All training on cloud GPUs (in use: Kaggle, 2x Tesla T4; images live in a PRIVATE Kaggle dataset). Copying thousands of PNGs with `Copy-Item` stalled silently for 7+ minutes: zip the folder or upload manually instead.

---

## 0. Current status (snapshot: 2026-09-19; refresh at every milestone)

**Phase.** The Vision lane has finished the *baseline* block (H2-H6 in section 9) and is already inside the *H9-H15 hardening* block: the stress suite is done and the leave-one-family-out novelty experiment is running. The Process and Systems lanes are behind: **the CP2 vertical slice (section 9, ~H9) does not exist yet.** Actual H-hour is not recorded (*fill in*). CP2/CP3 times will not be provided by the organizers, so treat CP2 as imminent and keep a demoable state at all times.

**Overall progress: about a quarter (~25%).** This is an estimate: section 2 effort split x a read of each lane. Numbers below come from results pasted from Kaggle runs; fold 0 only, one seed.

| Lane | Weight | Done | Not yet | Est. |
|---|---|---|---|---|
| Vision (M1, M2, validation) | 45% | Splits + locked holdout; EfficientNet-B0 baseline (fold 0); leak/shortcut diagnostics; stress suite; corruption augmentation | Folds 1-4 (out-of-fold), anomaly track, open-set result (running), fusion, calibration + conformal, cost-based decision, **localization (0 of 10 marks started)**, TTA | ~40% |
| Process and economics (M3, M4, M5) | 30% | Loaders + flow v0 code written (run output not seen) | Station parameter table, triangulated bottleneck, loss waterfall, surrogate/SimPy, root cause, profit model | ~10% |
| Systems (repo, README, contract, M6-M8, demo) | 25% | CP1 README corrected against the real data; Kaggle pipeline working; repo skeleton (folder command given, not confirmed) | Artifact contract, dashboard, recommendations, narrator, `--stage all`, model card | ~15% |

**Key results so far (fold 0, group-aware split):**
- Clean accuracy and macro-F1 1.000 on the validation fold and on `cond_holdout` (unseen brightness extremes); normal-vs-defect AUROC 1.000.
- Shortcut/leak checks: trivial global stats reach 0.638 (val) / 0.400 (cond_holdout) vs 0.20 chance; near-duplicates to train at cosine >= 0.995: 0.5% / 0.2%. No gross brightness shortcut or leak found; residual risk (crop/rotation near-copies) not excluded.
- Baseline model collapsed under noise (sigma 10: 0.345), JPEG q20 (0.276), blur sigma 3 (0.282), low-res 4x (0.497). With corruption augmentation every test is >= 0.98, including families not trained on (motion blur 0.998, speckle 0.998, noise sigma 40 0.980); clean cond_holdout 0.9994.
- Locked holdout (1,457 images) untouched.

**Marks at risk.** Localization (10) has no work and no location ground truth (image-level labels only); classification is saturated so it will not separate us from other teams. The differentiators left are novelty flagging, calibration/abstention, localization, and the linked flow/profit story.

**Next actions, in order:**
1. Read the novelty result (running) and log it for the Trust page.
2. **Build the CP2 vertical slice**: fold-0 predictions -> artifact contract -> minimal Streamlit page -> flow v0 -> a *simulated, labeled* quality-to-flow coupling.
3. Run folds 1-4 with augmentation (out-of-fold predictions), then calibration + conformal + cost-based decision.
4. Localization: Grad-CAM++ and a patch-kNN anomaly map, stated as heatmap-based.
5. Organizers confirmed no further information will come, so everything in section 4 stays unanswered: proceed on the assumptions in D10/D18 and disclose them in the README and on the Trust page.

---

## 1. The problem in plain words

A factory line has several stages, several product variants, and conditions that drift (lighting, batches, tooling). The organizers give three kinds of data: **inspection** (product images, probably with labels), **production** (stations, cycle times, WIP, downtime, changeovers, scrap, rework, batches, process settings) and **economics** (prices, costs, margins).

We build ONE software system that answers seven questions, and answers them *together*:

| # | Question | Output |
|---|---|---|
| Q1 | Is this unit OK or defective, and which defect family? | verdict + family |
| Q2 | Where is the defect? | box / mask / heatmap (only where data supports it) |
| Q3 | How sure are you, and have you seen this before? | calibrated confidence; UNCERTAIN / NOVEL flag instead of a forced guess |
| Q4 | What process condition likely caused it? | evidence-backed link between defect patterns and batch/process data |
| Q5 | Where is the flow constrained? | bottleneck station + evidence |
| Q6 | What does that cost in throughput and money? | throughput loss, cost of quality, margin |
| Q7 | What should change, and what happens to profit? | ranked advisory recommendations (simulated) |

**Explicitly NOT wanted:** another image classifier, a labeling tool, an isolated KPI dashboard. **Forbidden/unneeded:** live camera, PLC, robot, any hardware. Every recommendation and profit number must be labeled simulated/advisory.

**Wording traps in the brief (each is a scoring hook):**
- "flag uncertain or novel defect types instead of forcing a guess" → open-set detection + abstention (Explainability 5, Robustness 10).
- "localize defects where the data supports it" → if there are no box/mask labels, do NOT invent boxes; produce heatmaps and say the localization basis.
- "mixed product variants / changing inspection conditions / batch-to-batch drift" → the hidden test will shift on exactly these. Validation must mimic it.
- "recurring defect families" → cluster/aggregate at family level for root cause.
- "continuously updated view" → streaming replay in the dashboard.
- "unified decision-support system" → every screen links to the others (defect → batch → cause → station → dollars).

---

## 2. Where the marks are

| Checkpoint | Marks | What is really rewarded | Our move |
|---|---|---|---|
| CP1 README | 15 (5/5/5) | Problem understanding, architecture, approach | Write in H1–H2 using the rubric's own words as headings; specific to the actual data |
| CP2 Partial execution | 25 | Features + problem relatability | End-to-end thin slice touching all 4 pillars by ~H9; speak factory language (scrap, OEE, cost of escape) |
| CP3 Detection/classification | 15 | Accuracy on unseen, false rejects controlled | Group-aware validation, strong backbone, TTA |
| CP3 Localization | 10 | Box/mask/heatmap precision on unseen samples | Right method per label type; report IoU / Dice / PRO |
| CP3 Robustness | 10 | Lighting / orientation / batch / novel variants | Stress suite + augmentation + leave-variant-out |
| CP3 FR/FA handling | 5 | Missed defects vs unnecessary rejects | Economics-driven 3-way decision (accept/reject/review) |
| CP3 Root-cause | 5 | Validity of defect↔process links | Stratified, FDR-corrected, drift-aligned evidence cards |
| CP3 Explainability & confidence | 5 | Evidence, calibrated confidence, limitations | Temperature scaling + conformal sets + Trust page |
| CP3 Technical | 5 | Architecture, reproducibility, runtime reliability | Batch pipeline → artifacts → thin UI; one-command rerun |
| CP3 UI/UX | 5 | Operational clarity, trends, evidence | Streamlit + Plotly, linked pages, live replay |

**Reading of the table:** 40 of the 60 CP3 marks are vision measured on unseen data, plus 5 for calibration. So the *validation design* matters more than the model architecture: a leaky split gives a beautiful number that collapses on the hidden set. The bottleneck and profit engines are not itemized at CP3, but they are the spine of the "unified" claim that CP1/CP2 and the demo are judged on. Build them lean but correct and visibly connected. Do not gold-plate them at the expense of vision.

**Effort split:** Vision ≈ 45% · Root-cause + flow + profit ≈ 30% · Integration/UI/docs/repro ≈ 25%.

---

## 3. Winning thesis

> One closed decision loop: pixels → verdict with calibrated uncertainty → defect family → process cause → flow constraint → dollars → advisory action. Every arrow is measured, every output carries evidence, confidence and an "ADVISORY / SIMULATED" tag.

Six differentiators (each maps to marks):
1. **Group-aware validation + published stress-test table** (lighting, orientation, batch, variant). → Robustness 10, Accuracy 15.
2. **Hybrid supervised + anomaly detector with conformal abstention**, plus a leave-one-defect-family-out experiment proving novel defects get flagged. → Robustness, Explainability.
3. **Economics-driven decision threshold**: the OK/defect cut-off is chosen by minimizing expected cost (false-reject cost vs escape cost vs review cost), with a slider. → FR/FA 5, ties vision to profit.
4. **Root cause with rigor**: stratified by variant, BH-FDR corrected, change-point aligned with drift, effect sizes, "association not causation" labeling, negative results reported. → Root-cause 5.
5. **Digital-twin what-if**: SimPy line model validated against historical throughput/WIP, then used to rank interventions by Δmargin with intervals. → CP2 relatability, demo.
6. **Streaming replay dashboard** with linked pages. → UI 5, "continuously updated".

---

## 4. First 15 minutes: ask the organizers (write answers into the Decisions Log)

**Outcome: organizers confirmed no additional information will be provided. None of the questions below will be answered; see D10 and D18.**

1. Is there a hidden/unseen test set? Submission format for predictions (labels, boxes, masks, heatmaps)? Release time? Which metrics and IoU thresholds?
2. Exact times of CP1 / CP2 / CP3 and what is submitted at each (repo, README, demo, video, live run?).
3. Are pretrained weights, public datasets and LLM APIs allowed?
4. Can the data go to cloud GPUs (Colab / Kaggle / Lightning)?
5. Is there a data dictionary? Do inspection, production and economic tables share keys (unit_id, batch_id, timestamp, station)?
6. Will judges run our code or only watch a demo? On what machine (CPU-only?)?
7. Any organizer-provided compute credits?

Do in parallel: create the git repo, download pretrained weights NOW (venue Wi-Fi can degrade): DINOv2-small/base, a YOLO checkpoint, ConvNeXt/EfficientNet. Cache them in a private cloud folder.

---

## 5. Data intake & triage (H0:15 – H1:00, whole team)

**Immediately, all data:**
- Read-only copy of raw data; checksum; never edit in place.
- **Provenance check** (lesson from AeroMesh, where a "raw" clip turned out to be a rendered flythrough and it was discovered only after a full pipeline run): look at 50 random images per class. Are they real photos or synthetic renders? Any burned-in text, watermarks, borders, black padding, identical backgrounds per batch? Are labels noisy?
- Near-duplicate detection (perceptual hash + embedding cosine > 0.98) → assign duplicates to the same group so they never straddle train/val.
- Clever-Hans test: train a quick logistic regression on frozen embeddings to predict `batch_id` / `lighting` from images. If that is easy, images carry a batch signature the model can exploit instead of the defect. Plan photometric normalization + augmentation.
- Defect pixel-size distribution → decides input resolution and whether to tile. Never downscale tiny defects away.

**Tabular:** keys, join cardinality, time index, units, missingness, per-station columns, batch/variant/shift fields, economic parameters (price, unit cost, scrap cost, rework cost, escape/warranty cost, downtime cost per hour, changeover cost, inspection/review cost). List every economic assumption we must invent (mark them as assumptions in the UI).

**Decision tree by label type:**

| What we get | Vision plan | Localization metric |
|---|---|---|
| Boxes | YOLO detection (+ classifier head or second-stage crop classifier) | mAP50 / mAP50-95 |
| Masks | YOLO-seg or U-Net/SegFormer with pretrained encoder; derive boxes | Dice / IoU |
| Image-level labels only | Fine-tuned ConvNeXt/EfficientNet (or DINOv2 fine-tune); localize with Grad-CAM++ plus anomaly maps; state "heatmap-based" | pointing-game hit rate, pixel-AUROC if any masks exist |
| Mostly OK images, few defects | Anomaly-first (patch-feature kNN), then defect-family classifier via prototypes/kNN on the few labeled | pixel-AUROC, PRO |
| Tiny defects / large images | Tiled inference (overlapping crops) then merge | as above |
| Heavy imbalance | Class-balanced sampling or focal loss, then RE-CALIBRATE after | AUPRC, per-class recall |

---

## 5A. What the data actually turned out to be (verified; update if organizers correct us)

**Production data** (Mendeley "Manufacturing Data Shared Facility - Discrete-Event Simulation", Arena, CC BY 4.0): three CSVs. Model 1 is 3,000 x 10 and Model 2 is 3,000 x 17 (one replicate per row at a swept `Demand` 1-20). Model 3 is 605,620 x 78 (~20 stations, no swept parameter; whether its rows are replicates at one snapshot or a time series is **still unresolved**, check `Time_Now`). **None of the files has cost, price, profit, scrap, rework, defect, downtime, changeover, batch, lot or timestamp fields.** Flow/bottleneck layers are grounded in real simulation data; every quality-to-flow and profit coupling is a labeled assumption.

**Inspection data** (organizer images): 12,000 PNGs, 5 classes (crack, hole, normal, rust, scratch), exactly 2,400 each, class-per-folder labels only. As far as the scripts show there are **no boxes or masks and no batch/lighting/variant metadata**. Splits are therefore built from image similarity, not batch. Realized split: dev 8,741 (5 folds), locked holdout 1,457, `cond_holdout` 1,802.

**Consequences for the plan:**
- Localization is heatmap-based only and cannot be scored with IoU/Dice/PRO unless organizers supply masks.
- Batch/variant robustness cannot be measured from the data; we measure lighting (`cond_holdout`) and synthetic corruptions and say so on the Trust page.
- Artifact-contract fields `batch_id`, `variant`, `ts`, `station` do not exist for images: null them or generate labeled-simulated values (decision D17, open).
- Root cause (M3) has no real process-to-defect link to mine; it will run on simulated coupling and must be labeled that way.

---

## 6. Architecture

```
 raw data ─► [M0 ingest + splits + dedup] ─► parquet/feature store
                    │
   ┌────────────────┼──────────────────────────────┐
   ▼                ▼                               ▼
 [M1 Vision]     [M3 Root cause]              [M4 Flow + bottleneck]
 det/seg/cls      batch table, drift,           station metrics, visits,
 + anomaly maps   tests, GBM+SHAP,              utilization, WIP, waterfall,
 + open-set       evidence cards                SimPy digital twin
   │                │                               │
   ▼                │                               ▼
 [M2 Calibration +  │                          [M5 Profit model + what-if grid]
  cost-based        │                               │
  decision]         └──────────────┬────────────────┘
   │                               ▼
   └──────────────────► [M6 Recommendation engine + evidence cards (+ optional LLM narrator)]
                                   ▼
                    artifacts/ (parquet, json, png)  ◄── contract locked by H3
                                   ▼
                     [M7 Streamlit dashboard, streaming replay]
```

**Design principle:** heavy work is a batch pipeline that writes artifacts; the UI only reads artifacts. The dashboard never needs a GPU, internet or live model at demo time. (AeroMesh lesson: demo must survive with Wi-Fi off.)

**Artifact contract (lock by H3, then never break it):**
- `predictions.parquet`: unit_id, image_path, batch_id, variant, condition, station, ts, p_defect, class_probs, pred_family, decision (ACCEPT/REJECT/REVIEW), flag (NONE/UNCERTAIN/NOVEL), anomaly_score, conformal_set, heatmap_path, boxes/mask ref, localization_basis (box|mask|heatmap)
- `batch_metrics.parquet`: batch_id, ts, variant, n_units, defect_rate per family, FPY, process parameters, drift flags
- `evidence_cards.json`: id, family, hypothesis, variables, effect size, p_adj, CV lift, lag, stability across variants, confidence tier, caveat, verify_action
- `flow_metrics.json`: per station ρ, effective capacity, WIP, downtime %, changeover %, scrap %, rework %; bottleneck + evidence list; loss waterfall
- `sim_results.parquet`: scenario_id, parameters, throughput (mean, CI), WIP, margin (mean, CI)
- `profit_model.json`: model type, features, out-of-sample metrics, interval method
- `calibration.json`: ECE, reliability bins, risk–coverage, conformal q̂, operating threshold, cost table
- `robustness.csv`: perturbation × metric
- `recommendations.json`: ranked actions with ΔTH, Δmargin (CI), evidence ids, tier, assumptions

**Repo skeleton:**
```
neurax/
  README.md  MASTER_PLAN.md  MODEL_CARD.md  Makefile  requirements.txt  config.yaml
  src/ingest/  src/vision/  src/calib/  src/rootcause/  src/flow/  src/profit/  src/recs/
  app/ (Streamlit pages)   artifacts/   notebooks/   tests/ (smoke tests)
  run_pipeline.py --stage {ingest,vision,calib,rootcause,flow,profit,recs,all}
```

---

## 7. Module specs

### M1 — Vision (biggest lever; ~45% of effort)
- **Track A (supervised):** Ultralytics YOLO (det or seg) or a fine-tuned ConvNeXt/EfficientNet, chosen by the decision tree. Input size from the defect-size distribution. TTA (flips / rot90 / light photometric), average logits.
- **Track B (novelty):** frozen DINOv2 (or WideResNet50) patch features; memory bank of OK patches per variant (coreset-subsampled); kNN distance → pixel anomaly map + image anomaly score (PatchCore-style; `anomalib` is an option if it installs cleanly, do not burn time on it).
- **Track C (open-set):** distance to per-family prototypes in embedding space, max-logit/energy.
- **Fusion:** small gradient-boosted or logistic meta-model on [p_defect, class probs, anomaly score, heatmap stats, per-variant z-scores] → {OK, DEFECT(family), UNCERTAIN, NOVEL}. Trained ONLY on out-of-fold predictions.
- **Robustness:** augmentation that mimics the stated shifts (brightness/contrast/gamma, shadow gradients, color temperature, blur, noise, JPEG, rotations/flips, perspective, scale), plus per-variant/per-batch normalization if the Clever-Hans test flags a batch signature.
- **Localization output** always includes `localization_basis`. Heatmap threshold = F-optimal on validation.

### M2 — Calibration + cost-based decision
- Temperature scaling (or isotonic) fitted on a held-out *group* fold, never on train.
- **Split conformal prediction:** score s = 1 − p(true class); q̂ = the ⌈(n+1)(1−α)⌉/n quantile of calibration scores; prediction set = {y : p_y ≥ 1 − q̂}. Abstain (UNCERTAIN) when the set is empty or contains both OK and a defect class.
- **Three-way expected-cost decision** with calibrated p = P(defective):
  - ACCEPT cost = p · c_escape
  - REJECT cost = (1 − p) · c_false_reject
  - REVIEW cost = c_review
  - Pick the minimum. With no review option, reject iff p > c_FR / (c_FR + c_escape). Costs come from the economics data; any invented value is flagged as an assumption in the UI.
- Report: ROC, PR, FRR/FAR at the chosen operating point, cost-vs-threshold curve, ECE, Brier, reliability diagram, risk–coverage (AURC).

### M3 — Root-cause engine
1. Assemble batch/unit table: predicted (and labeled, for validation) defect rate per family per batch/station/variant/shift, joined to process parameters, supplier lot, tool wear, changeovers, timestamps. Exclude any feature measured after inspection (leakage).
2. Control charts (p-chart / CUSUM) on defect rate per family → drift windows.
3. Change-point detection (`ruptures`, PELT/BinSeg) on process parameters; cross-correlate with defect-rate shifts over ±k batches to estimate lag.
4. Stratified association tests (within variant), effect sizes (odds ratio / Cliff's δ), Benjamini–Hochberg FDR.
5. Multivariate: LightGBM predicting family occurrence from process parameters with batch-grouped CV; permutation importance, SHAP, partial-dependence to extract threshold statements ("risk rises above X").
6. Image-derived features per batch (defect location/size/orientation distributions) → tooling/fixture hypotheses.
7. **Evidence card** per hypothesis: variables, effect size, adj. p, CV lift, lag, stability across halves/variants, tier (High/Med/Low), caveat "association, not proven causation", how to verify.
- Validity guards: confounding by variant/time/shift, multiple testing, small n, shuffled-label permutation baseline. Report "no process link found for family X" when true.

### M4 — Flow & bottleneck
Per station s (m_s machines, cycle time CT_s, availability A_s, changeover loss fraction CO_s):
- Visits per good unit V_s (accounts for scrap upstream and rework loops, which consume capacity).
- Max sustainable good throughput = min over s of [ m_s · A_s · (1 − CO_s) / (V_s · CT_s) ].
- Utilization at demand λ: ρ_s = λ · V_s · CT_s / (m_s · A_s · (1 − CO_s)).
- **Bottleneck call requires triangulation:** highest ρ, queue/WIP piling up upstream, starvation downstream (and blocking upstream). Show all three as evidence. Recompute per variant because the constraint can shift with mix.
- Loss waterfall: theoretical → bottleneck capacity → downtime → changeover → scrap → rework → good units/h. OEE = availability × performance × quality. Little's law WIP = TH × CT for sanity checks; Kingman (VUT) for queue delay.
- **Cost of the constraint** (Theory of Constraints): an hour lost at the bottleneck is an hour lost for the whole line, valued at throughput margin per hour.
- **Digital twin (SimPy):** stations, buffers, stochastic cycle times (fit lognormal/gamma), time-to-failure/repair (exponential/Weibull), variant mix, changeovers, yield/rework. **Validate against history first** (simulated vs observed throughput/WIP within a stated tolerance) before any what-if. Scenarios: +capacity at bottleneck, −20% changeover, −30% defect rate for family F, earlier inspection stage, batch-size/mix change, buffer sizes. ≥200 replications → confidence intervals.
- If production data is only coarse aggregates: fall back to the analytical TOC/queueing model + Monte Carlo and say so.

### M5 — Profit model
- Accounting identity from the economics data: margin/h = good units/h × (price − variable cost) − scrap cost − rework cost − escape cost × FAR − downtime cost.
- ML cross-check (elastic-net or LightGBM, monotone constraints where sensible) on batch features (yield, defect rate by family, utilization, downtime, changeover, mix). **Time-based split** (train past, test future) because drift is the point. Conformalized/quantile prediction intervals.
- Pick whichever is better out of sample; show a margin waterfall. What-if margins come from sim outputs pushed through the chosen model.

### M6 — Recommendations
- Action catalog: (a) process setpoint corrections from evidence cards, (b) capacity/flow moves from M4, (c) inspection-policy moves from M2 (operating point, review budget, inspection placement), (d) mix/scheduling moves (batch variants to cut changeovers).
- Score = expected Δmargin (with 5–95% interval) × evidence-confidence tier; effort is an explicit assumption. Each recommendation shows why, expected impact, confidence, what could invalidate it, how to verify. All tagged ADVISORY.
- **LLM as narrator only** (Gemini or Claude API): input is the structured JSON; output cites evidence ids; it never computes numbers. Template fallback when offline or the API fails.

### M7 — Dashboard (Streamlit + Plotly; contract-first)
Pages, all reading `artifacts/`:
1. **Line Health** — live replay ticker (`st.fragment(run_every=…)` over time-ordered events): FPY, OEE, throughput, margin/h, drift alerts.
2. **Inspect** — unit viewer: verdict, family, overlay (box/mask/heatmap), confidence, conformal set, UNCERTAIN/NOVEL flag with reason, nearest known examples.
3. **Root Cause** — batch drift timeline + evidence cards.
4. **Flow & Bottleneck** — station utilization strip, WIP bars, waterfall, bottleneck marker with its three evidence lines.
5. **Profit & What-if** — sliders (threshold, defect reduction, capacity, changeover) over the precomputed sim grid → margin with interval.
6. **Trust** — calibration plot, risk–coverage, robustness table, leave-family-out novelty result, limitations, assumptions list.
- Cross-links everywhere (click a defect → its batch → its cause → its station → dollars). This is what stops it being "an isolated KPI dashboard".
- Global ADVISORY/SIMULATED badge. Clean dark or light theme, no marketing fluff.
- Fallback only if a strong Next.js person is free: build against the same JSON contract; otherwise Streamlit.

### M8 — Reproducibility
Pinned `requirements.txt`, seeds everywhere, config-driven, `run_pipeline.py --stage all` rebuilds all artifacts from raw data, CPU-capable inference path, weights via a download script (link), smoke tests, model card with limitations, tested from a clean environment.

---

## 8. Validation protocol (the part that decides CP3)

1. **Never random splits.** Group by batch / variant / lighting session / date, with duplicates merged into their group. Use GroupKFold for development.
2. **Locked final holdout by H2:** whole batches + at least one whole variant + one lighting condition, untouched until model selection is final (at most 2–3 evaluations ever). It simulates the hidden test.
3. **Stress suite** applied to validation images: brightness, gamma, contrast, shadow gradient, color temperature, blur, noise, JPEG, rot 0/90/180/270, flips, scale. Output `robustness.csv` (perturbation × metric), then augment to close the worst gaps. This table is the evidence for the 10 robustness marks.
4. **Novelty experiment:** hide one defect family from training, show it is flagged NOVEL/UNCERTAIN (report recall and AUROC), and that accuracy on the accepted subset improves (risk–coverage).
5. **Metrics:** per-class P/R/F1, macro-F1, AUROC/AUPRC, FRR/FAR, expected cost · mAP50/50-95, Dice/IoU, pixel-AUROC + PRO · ECE, Brier, AURC.
6. **Root-cause checks:** shuffled-batch permutation baseline, first-half vs second-half stability, recovery of any planted effect if the data dictionary reveals one.
7. **Sim check:** simulated vs observed throughput/WIP before what-ifs.
8. The inference script must run on a fresh image folder in one command and emit the organizers' required format (confirm in §4).

---

## 9. 24-hour timeline

Lanes (default team of 3; if fewer, merge as noted):
- **V — Vision lead:** M1, M2, validation (highest marks; you own this).
- **P — Process & economics:** M3, M4, M5.
- **S — Systems:** repo, README, contract, M6, M7, M8, demo.
- 2 people: V; P+S with a minimal Streamlit. Solo: V → P (lean) → S, and use the cut list in §10.

| Time | Milestone |
|---|---|
| H0:00–0:15 | Organizer questions, repo, roles, weights downloading |
| H0:15–1:00 | Data triage (§5) by all; S drafts README skeleton |
| H1:00–2:00 | **CP1 README v1** (rubric words as headings). Freeze split protocol + locked holdout. Baseline training launched on cloud GPU |
| H2:00–6:00 | V: baseline supervised + anomaly baseline, first group-CV numbers. P: batch table, drift charts, first tests, station metrics, bottleneck v0. S: Streamlit skeleton on mock artifacts; **contract locked by H3** |
| H6:00–9:00 | **Vertical slice:** one unit → verdict + heatmap + confidence → its batch → cause card → its station's bottleneck → margin. All pillars real (crude). Rehearse this 3-min thread. **CP2 ready by ~H9** (adjust to the real time) |
| H9:00–15:00 | V: augmentation, stress suite, calibration, conformal, fusion, cost-based thresholds, novelty experiment; queue long GPU runs for the sleep window. P: root-cause rigor, SimPy build + validation, profit model, what-if grid. S: page polish, streaming replay, evidence cards, Trust page, LLM narrator + fallback |
| H13–H18 | Staggered 60–90 min naps while long GPU jobs run |
| H15:00–18:00 | Full-pipeline run from a clean environment (`--stage all`). **Feature freeze at H18** |
| H18:00–21:00 | Final model selection on locked holdout (once), final artifacts, UI polish, limitations, model card, README final, screenshots, backup demo video |
| H21:00–23:00 | Dress rehearsal + Q&A drill (§13). Test offline and on a second machine. Tag `v1.0`. Verify links from mobile data in incognito. **Submit by H22–23** |
| H23:00–24:00 | Buffer only. No new features after H21 |

**Compute:** Kaggle notebooks (background "Save & Run All" survives a closed browser; check your weekly GPU quota), Lightning AI (your notes say ~80 free GPU h/month; check what AeroMesh left), Colab free only as a third option (sessions drop). Every training run checkpoints every epoch to persistent storage and can resume. First run on a small subset to validate the pipeline before committing hours (AeroMesh lesson).

**Priority when time is short:** vision metrics > calibration/abstention > root-cause cards > bottleneck view > profit what-if > polish.

---

## 10. Cut list & kill criteria

- No box/mask labels → skip detectors; classifier + heatmaps only, and state it.
- Supervised baseline still broken at H6 → fall back to classifier + CAM + patch-kNN anomaly only.
- SimPy not validating by H14 → analytical TOC/queueing model + Monte Carlo.
- LLM narrator flaky → templates (already the fallback).
- Do NOT build: labeling tool, chatbot UI, real-time video, anything touching hardware/PLC, novel architectures from papers, sweeps beyond two rounds.

---

## 11. Mistakes Ledger (pre-populated pre-mortem; add real ones as they happen)

| # | Trap | Guard |
|---|---|---|
| 1 | Random split leaks near-duplicate frames → inflated metrics | Group splits + dedup |
| 2 | Downscaling kills tiny defects | Check size distribution; tile |
| 3 | Accuracy on imbalanced data | AUPRC, FRR/FAR, expected cost |
| 4 | Uncalibrated softmax is overconfident | Calibrate on held-out group fold |
| 5 | Meta-model trained on in-sample predictions | Out-of-fold only |
| 6 | Spurious root cause: variant/time confounding, multiple testing, post-inspection features | Stratify, BH-FDR, feature audit, "association" wording |
| 7 | Data isn't what it claims (AeroMesh: rendered "raw" clip) | Provenance check in first 45 min |
| 8 | Model reads lighting/borders/watermarks, not defects (like the black-mask blob in AeroMesh, where the model learned the mask literally) | Clever-Hans test, Grad-CAM audit, normalization |
| 9 | Long runs on flaky sessions with no checkpoints | Epoch checkpoints + resume |
| 10 | Demo depends on live GPU/internet | Precomputed artifacts; test with Wi-Fi off |
| 11 | Dashboard becomes isolated KPIs | Cross-links between all pages |
| 12 | Overclaiming (causal language, unmeasured precision) | Every number has source + caveat |
| 13 | LLM invents numbers | Narrator only; structured inputs; template fallback |
| 14 | Feature creep late | Freeze at H18 |
| 15 | Tuning on the locked holdout | ≤3 evaluations total |
| 16 | Unknown checkpoint times | Ask in §4; submit early |
| 17 | README quoted data facts from the dataset's own documentation (10/16/77 columns; a `TODO-CONFIRM`) | Automated inventory first; README now states 10/17/78 and the missing fields (happened, fixed before the jury) |
| 18 | Perfect 1.000 on fold and unseen-lighting images looked like a leak | Ran shortcut + near-duplicate + coarse-thumbnail diagnostics before believing it; no leak found, caveat kept (happened) |
| 19 | Baseline model fragile to noise / JPEG / blur (0.28-0.50 accuracy) despite perfect clean scores | Stress suite before quoting any number; corruption augmentation + held-out corruption families (happened, fixed) |
| 20 | No batch/lighting/variant metadata for images, so group-by-batch splits are impossible | Similarity-grouped splits + brightness-tail holdout; residual leak risk from crops/rotations stated (happened) |
| 21 | Copying thousands of PNGs in PowerShell stalled silently | Zip or manual upload to Kaggle (happened) |

---

## 12. Decisions Log (provisional → confirm/edit)

| ID | Decision | Status |
|---|---|---|
| D1 | Streamlit + Plotly UI reading precomputed artifacts | provisional |
| D2 | Batch pipeline → artifact contract (locked H3) | provisional |
| D3 | Group-aware splits + locked holdout. Realized as: near-duplicate groups (cosine >= 0.995 on 32x32 thumbnails), `cond_holdout` = darkest/brightest 7.5% per class, locked holdout ~1/7 (max 3 evaluations), 5 dev folds | firm, implemented |
| D4 | Hybrid supervised + anomaly + open-set, conformal abstention | firm |
| D5 | Three-way expected-cost decision using economics data | firm |
| D6 | SimPy digital twin, validated before what-ifs | provisional |
| D7 | LLM narrator only, with template fallback | provisional |
| D8 | Repo + data outside OneDrive | firm |
| D9 | Everything tagged ADVISORY/SIMULATED | firm |
| D10 | Organizer answers (§4) | **closed: organizers confirmed no further information will come.** Hidden-test format, mask/box availability, pretrained-weight and cloud rules, shared keys, economics sheet and CP2/CP3 times all stay unknown; we proceed on D18 |
| D11 | Track A = timm EfficientNet-B0 (pretrained), 256 px, grayscale to 3 channels, class-weighted CE, AdamW + OneCycle, mixed precision, fixed 12 epochs, last epoch kept (no selection on val) | firm |
| D12 | Corruption augmentation (noise, blur, JPEG, low-res) is mandatory in training; robustness is reported on corruption families the model did not train on | firm |
| D13 | Localization = heatmap-based (Grad-CAM++ + anomaly map); no boxes invented | firm |
| D14 | Production data is synthetic and has no batch/defect/cost fields; quality-to-flow and profit coupling are explicit, editable, labeled assumptions | firm |
| D15 | Kaggle dataset with the organizer images stays PRIVATE | firm |
| D16 | Report each metric with its fold count and seed; single-fold numbers are labeled as such until folds 1-4 are run | firm |
| D17 | How to fill contract fields `batch_id`, `variant`, `ts`, `station` for images: null vs labeled-simulated | **open** |
| D18 | Assumptions register (unconfirmed; disclose in README and on the Trust page): pretrained ImageNet weights are allowed; cloud GPU use of the images is allowed (kept private); no hidden-test format known; no economics sheet; no join keys between inspection and production data; judges may either run the code or only watch the demo | provisional |
| D19 | Localization evidence without ground truth: heatmap faithfulness (deletion/insertion curves) plus a qualitative overlay grid; no IoU/Dice/PRO claimed | provisional |
| D20 | Hidden-test format unknown: `run_pipeline.py --stage infer --input <image folder>` writes a generic predictions CSV (file, verdict, family, calibrated confidence, flag, heatmap path) that can be reshaped quickly | provisional |

---

## 13. CP1 README skeleton, demo script, Q&A

**README headings (use the rubric's words):** 1 Problem Understanding (the seven questions in our words, constraints, what we deliberately don't build) · 2 Architecture (diagram, data flow, artifact contract) · 3 Approach (per module: method + why; validation protocol; uncertainty; economics coupling) · 4 Data (what was provided, assumptions) · 5 Metrics plan · 6 Status per checkpoint · 7 Limitations · 8 Reproduce in one command.

**CP2 relatability:** include a "day in the life of the quality engineer" walkthrough using real numbers from the data.

**3-minute demo:** (0:00) one-sentence problem + Line Health live · (0:25) a defective unit: verdict, heatmap, confidence · (0:55) an UNCERTAIN/NOVEL unit: why it abstained + leave-family-out result · (1:25) root-cause card on the drift timeline · (1:55) bottleneck with three evidence lines + waterfall · (2:20) what-if slider: threshold or defect reduction → margin interval · (2:45) Trust page: calibration, robustness table, limitations · close with hidden-test numbers.

**Likely judge questions (prepare crisp answers):**
1. How do you know it isn't overfit to a batch or lighting? → group splits, holdout, stress table, Clever-Hans test.
2. What happens on a defect you've never seen? → anomaly + open-set + conformal abstention; leave-family-out result.
3. Why should we trust the root cause? → stratification, FDR, drift alignment, stability, negative results, "association" labeling.
4. How do you decide the bottleneck? → visits-adjusted capacity + queue + starvation triangulation, per variant.
5. What does the profit model assume? → the assumptions list on the Trust page; time-based OOS test; intervals.
6. Why not just YOLO? → YOLO answers Q1/Q2 only; the brief asks Q1–Q7 as one system.
7. Runtime and reproducibility? → one command, CPU path, pinned deps, clean-env test.
8. What would you do with more time/data? → active learning on UNCERTAIN units, online drift adaptation, real cost data.

---

## 14. Progress Tracker

Legend: [x] done, [~] partly done or in progress, [ ] not started. Updated 2026-09-19.

- [x] Organizer answers logged (§4): none will be given (organizers confirmed); assumptions recorded in D10/D18
- [~] Repo + weights + cloud compute ready (Kaggle 2x T4 verified; EfficientNet-B0 weights download at runtime; DINOv2/YOLO not fetched; repo skeleton folders not confirmed)
- [~] Data triage done: label type (image-level, 5 classes) and shortcut/duplicate checks done; provenance (real vs synthetic photos) and defect-size distribution not confirmed
- [x] Split protocol + locked holdout frozen (1,457 locked images, untouched)
- [x] CP1 README submitted (corrected against the real data; confirm the submission itself)
- [ ] Artifact contract locked (see D17)
- [~] Baseline supervised + anomaly running, group-CV numbers logged (supervised: fold 0 of 5 only; anomaly track not started)
- [ ] Batch table + drift + first evidence cards (no batch data exists; needs simulated coupling)
- [~] Station metrics + bottleneck v0 (code written; run output and the Model 3 `Time_Now` result not reviewed)
- [ ] Dashboard skeleton on real artifacts
- [ ] Vertical slice demo-able -> CP2 submitted
- [x] Augmentation + stress suite + robustness.csv (fold 0; results in `stress_fold0_compare.csv` on Kaggle, still to be copied into `artifacts/robustness.csv`)
- [ ] Calibration + conformal + cost-based decision
- [~] Novelty (leave-family-out) experiment (scratch hidden, running; other families not yet run)
- [ ] Root-cause rigor (stratify, FDR, change-point, SHAP)
- [ ] SimPy validated vs history; what-if grid
- [ ] Profit model (time-split OOS) + intervals
- [ ] Recommendations + narrator (with fallback)
- [ ] Feature freeze (H18)
- [ ] Clean-env `--stage all` passes
- [ ] Final holdout evaluation (once)
- [ ] Model card, limitations, final README, backup video
- [ ] Offline test + second-machine test + links verified from mobile data
- [ ] `v1.0` tagged and submitted

**Tally:** 4 done, 5 in progress, 15 not started (of 24 items). Item counts overstate the vision lane's share of the marks, which is why section 0 uses the effort-weighted estimate instead.

---

## 15. Sponsor card (StepUpMark.AI) verdict

The card offers 50 sign-up credits, a 10% affiliate commission, and "unlimited chat with 25+ AI features with SEO-optimized outputs" from ₹99. It is a marketing/content-writing tool. Nothing in it helps with vision, tabular analysis, simulation or profit modeling, and SEO optimization earns no marks here. Don't spend time on it, and don't upload organizer data to it. Ignore the affiliate pitch. Better use of attention: ask the organizers whether any GPU/cloud credits exist (§4 Q7).

---

## 16. Formula cheat sheet

- Max good throughput = min_s [ m_s · A_s · (1 − CO_s) / (V_s · CT_s) ]
- ρ_s = λ · V_s · CT_s / (m_s · A_s · (1 − CO_s))
- OEE = Availability × Performance × Quality · Little: WIP = TH × CT · Kingman: W_q ≈ (ρ/(1−ρ)) · ((c_a² + c_s²)/2) · τ_s
- Reject iff p > c_FR / (c_FR + c_escape) (no review option); with review: choose min(p·c_escape, (1−p)·c_FR, c_review)
- Split conformal: q̂ = ⌈(n+1)(1−α)⌉/n quantile of {1 − p_true}; set = {y : p_y ≥ 1 − q̂}
- BH-FDR: sort p-values, reject up to the largest k with p_(k) ≤ (k/m)·α
