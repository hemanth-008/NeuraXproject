# NEURAX 3.0 - QualityTwin - Execution Plan v3 (REBUILT 2026-09-20 04:10 IST)

**Read this first.** The original `NEURAX_Execution_Plan_v3.md` lived only in an earlier chat's sandbox and could not be recovered. This file is a rebuild from `NEURAX_Handoff_Chat3.md`, `NEURAX_Master_Plan.md`, and Codex's Kaggle report. It is NOT the original text, and I have not run any of it. Where a task needs details I do not have (decisions D21-D28, the process-layer spec, UI spec), the prompt tells the agent to read `docs/NEURAX_GodTier_Plan_v2.md` and follow it, and to report any conflict rather than choose silently.

**Deadline: 08:30 IST on 2026-09-20 (stated by the user).** All clock times below are IST.

---

## 1. Clock and gates

| Clock | Gate / action |
|---|---|
| 04:10-04:30 | **H1: launch the full 5-fold run on Kaggle** (section 4). Start Codex cards C1, C3, C3L, C4, C6, C8 in parallel threads. Antigravity runs AG-2 (Inspect + Upload + Command Center). |
| 04:50-05:10 | Start C5 (needs C4), C7 (needs C6 and C5, fixtures allowed), C9 (needs C8). |
| ~05:00-05:30 | Expected end of the 5-fold run (estimate, section 2). **Hard cutoff 06:00: use whatever folds are complete.** |
| when 5-fold ends | H2: launch the LOFO novelty job (needs C3L notebook), about 20-30 min. |
| 05:30-06:15 | A6: ingest Kaggle outputs, first full pipeline run. A3/A4 finish pages. |
| 06:15 | G1b: LOFO in, otherwise the Trust page says "novelty: not run for N of 4 families" (do not show old numbers as current). |
| **06:30** | **G2 FEATURE FREEZE.** Clean-venv `run_pipeline.py --stage all` passes, tag `rc1`. No new features after this. |
| 06:30-07:00 | **H3: locked holdout, once**, by hand. A5 visual QA in parallel. C9b final README + model card. |
| 07:00-08:00 | H4: rehearsal x2, backup video, offline test, second machine, tag `v1.0`. |
| 08:00-08:30 | Submit by 08:00-08:15. Last 15 min is buffer only. |

**Kill rules:** twin MAPE > 25% -> analytical + Monte Carlo. Planted-effect recovery fails -> fix the engine, never loosen the plant. An agent change that breaks contract tests -> revert it. **Cut order (first to go):** variant-mix slider and any LLM narrator -> U-Net -> Model 3 triangulation -> tiling + domain probe -> ML profit cross-check -> Demo Mode. **Never cut:** the full-augmentation 5-fold run, `infer` + Upload page, an honest Trust page, `--stage all`, the holdout run, the backup video.

---

## 2. Facts as of 04:10 IST

- **Kaggle:** private notebook `neurax-train-5fold` exists (imported by Codex), dataset `h3m4nth/dataset001` at `/kaggle/input/datasets/h3m4nth/dataset001`, accelerator T4 x2, torch 2.10.0+cu128, timm 1.0.26. Imported script sha256 matches the local `notebooks/train_5fold.py` (`26e6f724...2522524`). Attached `splits.csv` matches the embedded one after excluding the locked rows. DRY_RUN passed for folds 0 and 1 with every expected output produced. The full 12-epoch run has NOT been started. Weekly GPU quota is NOT verified.
- **Throughput:** fold 1 of the dry run measured 309.13 img/s; fold 0 measured 7.17 img/s (warm-up). Both came from a ~200-image, 1-epoch dry run, so treat them as indicative only. Back-of-envelope: about 7,000 training images x 12 epochs is about 84,000 image-passes per fold, which is roughly 4.5 min of training at 309 img/s, plus embeddings and the stress suite. Plan for 10-15 min per fold, 30-45 min per GPU for three folds. **Unmeasured until the first epoch logs appear. If one epoch takes more than 60 s on a fold, tell me.**
- **The old fold-0 model no longer exists.** The reference notebook was a draft that was never re-executed after the Kaggle disconnect; there are no weights anywhere. The earlier numbers (novelty AUROC 0.991, deletion/insertion 0.397/0.826, ECE 0.0006, T = 0.05) came from that vanished run. They must be recomputed on the new ensemble. Do not present them as current.
- **`artifacts/predictions.parquet` (CP2 version):** 3,550 rows = 1,748 `val_fold0` + 1,802 `cond_holdout` (locked holdout absent). 712 ACCEPT / 2,838 REJECT / 0 REVIEW; 2 UNCERTAIN (`normal_02075` ACCEPT, `rust_00187` REJECT). `anomaly_score` and `heatmap_path` are null for every row.
- **Repo:** Antigravity untracked the images, wrote `requirements.txt`, `AGENTS.md`, module skeletons, `docs/` (Master and GodTier plans), a 7-page Streamlit shell and design system. History still contains the images: make the GitHub repo private (section 4, H0).

---

## 3. SHARED RULES block (prepend to EVERY agent prompt)

```
SHARED RULES
You work in C:\NeuraxHack (Windows, PowerShell; Python is .venv\Scripts\python.exe). Other agents edit this folder at the same time.
- Edit only the paths listed under OWNS in your task. Read AGENTS.md and docs/ first.
- Git: stay on the current branch. Never run git add -A, git add ., stash, reset, clean, checkout, switch, rebase or push. Commit explicit paths only: git add <paths> then git commit -m "<msg>" -- <paths>. If .git/index.lock exists, wait 5 seconds and retry.
- Do not install or upgrade packages. If something you need is not importable, say so, use the stated fallback, and continue.
- Never read, score, evaluate on or reference locked-holdout rows (splits.csv split == 'holdout'). Assert this in code wherever you load data.
- Anything the app or the infer path runs must be CPU-only and OFFLINE.
- Never invent numbers or claim something works without a run log. Every metric you report was printed by a command you ran; every metric inside an artifact was computed by code, never typed.
- Anything that depends on an assumption carries its id (D14, D18, D22, D24, D25, economics.*) and, if simulated, the tag ADVISORY/SIMULATED. Wording: "association, not proven causation".
- Final report: files changed, commands run and their outcomes, what you could NOT verify.
```

---

## 4. Human steps

**H0 (1 min, now): make the GitHub repo private.** GitHub -> repo `hemanth-008/NeuraXproject` -> Settings -> General -> Danger Zone -> Change visibility -> Make private. Untracking `data/` does not remove the organizer images from the public history, and D15 says the data stays private.

**H1: launch the full 5-fold run (target 04:30).**
1. Kaggle -> open notebook `neurax-train-5fold`.
2. Check the GPU quota shown in the notebook's session/settings panel. You need roughly 1.5 h of GPU time; if under 3 h remain, tell me before launching.
3. In the launch cell change `DRY_RUN=True` to `DRY_RUN=False`. Change nothing else. Keep Internet ON (the pretrained weights download; it was on for the dry run).
4. Run mode. **Recommended: Save Version -> Save & Run All (Commit).** It keeps running if the browser closes or the session drops (your earlier disconnects wiped `/kaggle/working`), but no files are visible until it finishes. Alternative: interactive Run All with the tab open; then download each fold's files as "DONE fold k" prints, because a disconnect wipes everything.
5. Note the start time. Expect the first fold to finish within about 15 min; report the first epoch time.
6. When it finishes: Output tab -> download all -> unzip to `C:\NeuraxHack\kaggle_downloads\run1\` (gitignored). Tell Antigravity the folder name.

**H2: LOFO novelty run.** After C3L has produced `notebooks/lofo_novelty_kaggle.ipynb`, ask Codex: "Do C2-K Phase 4 for notebooks/lofo_novelty_kaggle.ipynb: import as a NEW private notebook named neurax-lofo, attach the same private dataset, GPU T4 x2, run DRY_RUN=True, report, then stop." Then you set DRY_RUN=False and run it (same commit-vs-interactive choice as H1). Download the outputs into `kaggle_downloads\lofo\`.

**H3: locked holdout, ONCE, after the 06:30 freeze.** From `C:\NeuraxHack`, run this one line, and paste me the output before it goes anywhere else:
```
.venv\Scripts\python.exe scripts\eval_locked_holdout.py --confirm
```
Do not tune anything after seeing it. If it fails to run, fix the failure, not the model. The script counts runs and refuses at 2 or more without `--override`.

**H4: final.** Rehearse the 3-minute demo twice on the presenting machine. Offline test (Wi-Fi off, fresh terminal, `streamlit run app.py`). Repeat on a second machine. Record a backup video of the full demo. Tag `v1.0`. Confirm the repo link works from your phone on mobile data (incognito). Confirm the repo is private or that the organizers accept the visibility you chose. Submit by 08:15.

---

## 5. Codex cards

Prepend the SHARED RULES block to each. Launch in separate threads: **now C1, C3, C3L, C4, C6, C8; then C5, C7, C9 as their inputs land; C9b at the freeze.**

### C1 - Artifact contracts, validators, fixtures
```
OWNS: contracts/, scripts/validate_artifacts.py, scripts/make_fixtures.py, src/common/, tests/fixtures/, tests/test_contracts.py, and ONLY the economics: and sim: blocks of config.yaml.
Read docs/NEURAX_Master_Plan.md section 6 (artifact contract) and 5A, and AGENTS.md.
1. Write contracts/<artifact>.schema.json (JSON Schema 2020-12) for every artifact: predictions (parquet columns + dtypes: the columns of the current artifacts/predictions.parquet plus OPTIONAL box_xyxy, novel_score, top2, tile_boxes, mixed_defect_flag, ensemble_agreement), batch_metrics, replay_stream, sim_unit_map, sim_truth, evidence_cards, rootcause_recovery, flow_metrics, twin_validation, sim_results, profit_model, recommendations, calibration, novelty, localization, robustness_orient (csv columns), knn_bank (npz keys), status, holdout_result, holdout_runs. Required fields follow Master Plan section 6; anything not specified there is optional. additionalProperties true. Every JSON artifact carries meta {generated_at, source, assumption_ids[], tag}.
2. src/common/config.py: load_config() reading config.yaml; economics() returning c_escape, c_false_reject, c_review plus optional price, unit_cost, scrap_cost, rework_cost, downtime_cost_per_hour, each with its assumption id. Add missing keys to config.yaml economics as ASSUMED placeholders with comments.
3. scripts/validate_artifacts.py [--require-all] [--dir artifacts]: validates every artifact present (json by schema, csv by columns, parquet by columns and dtypes, npz by keys), prints a table (artifact, ok | missing | invalid, first 3 errors), exit 1 on any invalid (missing is not invalid unless --require-all). Do not edit existing artifacts; if artifacts/predictions.parquet or flow_v0.json violate a schema, report the exact violations.
4. scripts/make_fixtures.py writes tiny deterministic fixtures for every artifact into tests/fixtures/ (about 50 rows, ids prefixed FIXTURE_, meta.source = "fixture").
5. tests/test_contracts.py: fixtures validate; a mutated fixture (missing required field, wrong dtype) fails; the current artifacts/predictions.parquet validates.
Done when pytest -q tests/test_contracts.py passes and validate_artifacts.py runs. Report.
```

### C3 - Calibration, conformal, cost decision, kNN novelty bank, ensemble predictions
```
OWNS: src/calib/, scripts/build_calibration.py, tests/test_calib.py; writes artifacts/calibration.json, artifacts/novelty.json (the "knn" section only), artifacts/robustness_orient.csv, artifacts/knn_bank.npz, artifacts/predictions.parquet (rebuild).
INPUT: Kaggle outputs unzipped under kaggle_downloads/<run>/ with the names written by notebooks/train_5fold.py (read that script and notebooks/kaggle_facts.md for exact names and columns). Real files arrive in about an hour: build and test against the dry-run layout or a synthetic fixture with the same names, and the same script must run unchanged on the real files. Accept any subset of folds (1 to 5) and record in calibration.json which folds were used.
1. OOF assembly: concatenate oof_fold{k} for the folds present (each dev image is predicted only by the fold model that did not train on it). cond_holdout = mean of logits over the fold models present. Also load oof_shift_fold{k} (the seeded-perturbation predictions).
2. Temperature scaling on OOF logits: fit T by NLL over a log grid 0.05-10 plus refinement. If the fit lands on a bound or the NLL surface is flat, record t_bound_hit: true in calibration.json and keep the fitted T only if ECE improves; never clip silently. Report ECE (15 bins), Brier and NLL, raw vs scaled, separately for clean OOF, cond_holdout and the shifted set.
3. Split conformal, alpha = 0.10, with a group-aware calibration/test split of the OOF (use the group column of splits.csv): q_hat, empirical coverage on the held-out half, set-size histogram, for clean AND shifted data. UNCERTAIN when the set is empty or contains both normal and any defect class.
4. Decision per Master Plan M2 using economics() from src/common/config.py: three-way min-expected-cost decision; UNCERTAIN and NOVEL route to REVIEW (write this rule into calibration.json). Write the cost-vs-threshold curve, FRR/FAR at the operating point, expected cost per unit, and the risk-coverage curve (AURC), for clean OOF and for the shifted set (the set where errors exist).
5. kNN novelty on fold-0 backbone embeddings: known-class bank = bank_fold0 subsampled by coreset to <= 8,000 vectors (fp16, npz < 25 MB); score = mean cosine distance to the k=5 nearest bank vectors; threshold at 5% FPR on fold-0 val. Save artifacts/knn_bank.npz and put threshold, k and embedding source in novelty.json under "knn". Compute anomaly_score for every unit.
6. robustness_orient.csv: concatenate stress_fold{k}.csv (perturbation, acc, macro_f1, n, fold) plus a mean row per perturbation; flag any mean < 0.95.
7. Rebuild artifacts/predictions.parquet from the ensemble for ALL dev units + cond_holdout (never locked): unit_id (image file stem), image_path, p_defect, class_probs (temperature-scaled), pred_family, decision, flag (NONE/UNCERTAIN/NOVEL), conformal_set, anomaly_score, true_label, source_split, ensemble_agreement (fraction of fold models agreeing with the ensemble argmax; cond_holdout only, null otherwise). Leave batch_id, variant, condition, station, ts, heatmap_path, boxes_mask_ref, localization_basis null; other tasks fill them. First copy the old file to artifacts/predictions_fold0_cp2.parquet.
8. tests/test_calib.py on synthetic logits: temperature recovers a known T; conformal coverage >= 1 - alpha - 0.02; decision rule matches the formulas in Master Plan section 16.
Done when pytest passes and the script runs on the fixture. Report headline numbers exactly as printed.
```

### C3L - Leave-one-family-out novelty (all four families), script + notebook generator only
```
OWNS: notebooks/lofo_novelty.py, scripts/build_lofo_notebook.py, notebooks/lofo_novelty_kaggle.ipynb, tests/test_lofo_dryrun.py. Do NOT edit notebooks/train_5fold.py or scripts/build_kaggle_notebook.py; copy what you need.
1. notebooks/lofo_novelty.py, self-contained like train_5fold.py (same data loading, splits handling with an assertion that no locked row is read, augmentation, model recipe, AMP, resume, DRY_RUN, banner with sha256). For each family F in {crack, hole, rust, scratch}: train ONE EfficientNet-B0 on fold-0 TRAIN images excluding class F (EPOCHS configurable, default 8, use 12 if the measured speed allows), then score fold-0 val + cond_holdout known classes (label 0) versus images of F (label 1) with (a) 1 - max softmax, (b) energy score, (c) kNN embedding distance to the known-class train bank (k=5, cosine). Report AUROC and recall at 5% FPR per detector, accuracy on the accepted subset at the 5%-FPR threshold, and a risk-coverage curve. Write lofo_{F}.json per family and an aggregate novelty_lofo.json (per family, mean, min, max). Two families per GPU via subprocess (CUDA_VISIBLE_DEVICES), like the launcher in train_5fold.py.
2. scripts/build_lofo_notebook.py generates notebooks/lofo_novelty_kaggle.ipynb the same way scripts/build_kaggle_notebook.py does (embedded splits, env check, script source, launch cell with DRY_RUN=True default).
3. Local CPU dry-run test on a tiny subset: every output file exists and no locked id appears in any output.
4. Do NOT touch Kaggle. Stop after step 3 and give me the exact steps to import and launch it.
Report files, dry-run results, estimated GPU time per family (state the basis), what you could not verify.
```

### C4 - Simulated factory layer (labeled SIMULATED) + planted effects
```
OWNS: src/sim/, scripts/build_sim_layer.py, scripts/merge_sim_into_predictions.py, tests/test_sim_layer.py; writes artifacts/batch_metrics.parquet, artifacts/replay_stream.parquet, artifacts/sim_unit_map.parquet, artifacts/sim_truth.json.
Read docs/NEURAX_Master_Plan.md 5A and 7 (M3) and docs/NEURAX_GodTier_Plan_v2.md (decisions D24 and D25 and the process-layer section). Follow the docs where they specify details; if they conflict with this card, follow the docs and tell me.
WHY: the organizers' production files contain no batch, lot, process-parameter, defect or timestamp fields (D14). We build a LABELED-SIMULATED factory layer so root-cause and recovery can run. Image labels and model outputs are real; batch/process data are simulated. Every artifact carries meta.tag "ADVISORY/SIMULATED" and assumption ids D14, D24, D25.
1. Deterministic generator (seed in config.yaml sim: block): about 120 batches over about 5 simulated days at minute resolution, 3 variants, 3 shifts, and process parameters per batch (names from the docs) with realistic drift and one change point. Include confounders (variant and time both correlate with some parameters) so naive correlation misleads and stratification matters.
2. Planted effects per D24/D25: three families get a planted parameter->defect-rate effect (threshold or monotone form) mapped as in D25; crack is the deliberate NO-LINK control (the demo claim is "planted 3, found 3, correctly no link for crack"; if the docs say otherwise, follow the docs and report). Units are drawn from the scored pool (dev + cond_holdout rows of artifacts/splits.csv; NEVER locked) so each batch's true family mix follows the planted effects. sim_unit_map.parquet: unit_id (image file stem), batch_id, variant, station, condition, ts (all simulated).
3. batch_metrics.parquet per Master Plan section 6 (batch_id, ts, variant, n_units, defect_rate per family, FPY, process parameters, drift flags). Rates come from TRUE labels; the root-cause engine uses predicted families when available.
4. replay_stream.parquet: time-ordered unit events (unit_id, ts, batch_id, station, decision if artifacts/predictions.parquet exists else null) for the live-replay ticker.
5. sim_truth.json: the planted effects (family, parameter, form, size, station). meta states it is read ONLY by the recovery test.
6. scripts/merge_sim_into_predictions.py: idempotent, fills batch_id, variant, condition, station, ts in artifacts/predictions.parquet from sim_unit_map and touches no other column; keeps a .bak copy.
7. tests: determinism; each planted parameter truly shifts its family's rate in the generated data; crack independent of every parameter; no locked id in any output; confounding present (variant explains part of a decoy parameter).
Report the planted-effects table and artifact sizes.
```

### C5 - Root-cause engine + planted-effect recovery
```
OWNS: src/rootcause/, scripts/run_rootcause.py, tests/test_rootcause.py; writes artifacts/evidence_cards.json and artifacts/rootcause_recovery.json. Reads artifacts/batch_metrics.parquet, artifacts/sim_unit_map.parquet, and artifacts/predictions.parquet if present (else the fixtures in tests/fixtures/). Read docs/NEURAX_Master_Plan.md sections 7 (M3) and 11 and follow M3 steps 1-7.
1. Drift: p-chart / CUSUM on defect rate per family across batches; change-point detection on process parameters (ruptures only if importable, else implement CUSUM or binary segmentation yourself); cross-correlation lag between parameter changes and rate shifts.
2. Stratified association tests within variant, effect sizes (Cliff's delta or odds ratio), Benjamini-Hochberg FDR across all family x parameter tests.
3. Multivariate model per family with batch-grouped CV (LightGBM if importable, else sklearn GradientBoosting), permutation importance, a shuffled-batch permutation baseline, partial dependence for threshold statements ("risk rises above X").
4. Stability: first half vs second half of the timeline and across variants.
5. evidence_cards.json per hypothesis: id, family, hypothesis, variables, effect size, p_adj, CV lift, lag, stability, confidence tier (High/Med/Low), caveat ("association, not proven causation; labeled-simulated layer D24"), verify_action. Include explicit "no process link found" cards where true.
6. THE ENGINE MUST NOT READ artifacts/sim_truth.json. Only scripts/run_rootcause.py --recovery reads it, AFTER the cards are written, to write rootcause_recovery.json: planted effects, found effects (parameter and direction), matches, false positives, misses, and whether the no-link control stayed clean. If recovery fails, fix the engine; never loosen the plant.
7. tests: on a tiny synthetic batch table with a known planted effect the engine recovers it and a null family stays clean.
Report recovery counts exactly as computed, runtime, and what you could not verify.
```

### C6 - Flow, bottleneck triangulation, digital twin, scenario grid
```
OWNS: src/flow/, scripts/build_flow.py, tests/test_flow.py; writes artifacts/flow_metrics.json, artifacts/twin_validation.json, artifacts/sim_results.parquet (throughput and WIP columns only; C7 adds margin columns). Reuse src/ingest/loaders.py and scripts/flow_v0_report.py without editing them. Read docs/NEURAX_Master_Plan.md 5A, 7 (M4) and 16.
1. Models 1 and 2 (data/Model 1|2, swept Demand 1-20): per-station utilization, queue time and WIP vs demand; bottleneck call with THREE evidence lines (highest utilization, queue/WIP pile-up, starvation or blocking), each with its numbers. For Model 2 report that Drilling has the longer mean queue while Assembly has the higher utilization; disclose it, do not hide it.
2. Model 3 (data/Model 3, 605,620 x 78; Time_Now is constant, so a single snapshot): read memory-efficiently (usecols or chunks); per-station table (about 20 stations) with utilization, queue, WIP and the bottleneck with the same evidence lines. State that it is a snapshot.
3. Coupling assumption (D14): rho_new = rho_baseline x (1 + rework_rate), labeled ASSUMPTION. Loss waterfall per M4 where the data supports it and "not observable" where it does not (there are no downtime, changeover or scrap fields).
4. Digital twin: SimPy if importable, else a vectorized numpy Monte Carlo (do not install packages). Fit per-station service-time distributions from the data, >= 200 replications, confidence intervals. VALIDATE against observed throughput and WIP per demand level: twin_validation.json with MAPE and pass/fail at 25%. If MAPE > 25%, fall back to an analytical (M/M/c or Kingman) + Monte Carlo model and say so in the file.
5. sim_results.parquet: scenarios (rework rate, +capacity at the bottleneck, demand level) with throughput and WIP mean and a 5-95% interval.
6. tests: analytical utilization formula, Little's law sanity, twin deterministic given a seed.
Report every headline number exactly as printed, and what you could not verify.
```

### C7 - Profit model + recommendations
```
OWNS: src/profit/, src/recs/, scripts/build_profit.py, scripts/build_recs.py, tests/test_profit_recs.py; writes artifacts/profit_model.json, artifacts/recommendations.json, and ADDS margin columns to artifacts/sim_results.parquet (never drop existing columns). Reads config.yaml economics, sim_results.parquet, flow_metrics.json, evidence_cards.json, calibration.json. Read docs/NEURAX_Master_Plan.md 7 (M5, M6).
1. Profit: accounting identity margin/h = good units/h x (price - variable cost) - scrap - rework - escape cost x false-accept rate - downtime cost. ALL economic parameters come from config.yaml economics and are flagged ASSUMED (D18/D22): the organizers gave no economics. Margin waterfall, and margin with 5-95% intervals pushed through sim_results. The ML cross-check is on the CUT LIST: skip it.
2. Inspection-policy economics from calibration.json's cost-vs-threshold curve: expected cost per unit at the chosen operating point vs accept-all vs reject-all.
3. Recommendations per M6: catalog (a) process setpoint corrections from evidence cards, (b) capacity/flow moves from flow_metrics and sim_results, (c) inspection-policy moves, (d) scheduling. Each: id, action, rationale, expected delta throughput and delta margin with interval, evidence ids, confidence tier, assumptions, what could invalidate it, how to verify, tag ADVISORY/SIMULATED. Score = expected delta margin x tier weight. Template text only; no LLM calls, no network.
4. If an upstream artifact is missing, use tests/fixtures/ and set meta.source = "fixture" so it can never be mistaken for real output.
5. tests: identity arithmetic, ranking monotonic in score, every recommendation carries an assumption id and the ADVISORY tag.
Report exactly what ran.
```

### C8 - `infer` stage, Grad-CAM++, boxes, localization evidence
```
OWNS: src/vision/, run_pipeline.py (create it; C9 extends it), scripts/eval_localization.py, scripts/build_demo_heatmaps.py, tests/test_infer.py; writes artifacts/heatmaps/, artifacts/localization.json, and fills heatmap_path, box_xyxy, localization_basis in artifacts/predictions.parquet for a curated subset only. Read docs/NEURAX_GodTier_Plan_v2.md section 4 and decisions D13/D19/D26, and docs/NEURAX_Master_Plan.md M1/M2.
GOAL: run on ANY image folder, CPU-only, offline, one command: .venv\Scripts\python.exe run_pipeline.py --stage infer --input <folder> [--output <folder>].
1. src/vision/infer.py: load every artifacts/models/fold{k}.pt present (fp16 state_dict -> float32 CPU; timm efficientnet_b0 with pretrained=False; nothing may touch the network). Preprocess any PNG/JPG/BMP of any size or mode (RGB/RGBA/L -> grayscale -> 256x256 -> 3 channels). Ensemble = mean of logits (optional TTA flags: flips, rot90; default off). Apply temperature and conformal q_hat from artifacts/calibration.json, the cost decision from config.yaml economics, and kNN novelty from artifacts/knn_bank.npz (threshold from novelty.json) -> NOVEL; UNCERTAIN and NOVEL -> REVIEW. If weights or calibration.json are missing, exit non-zero with a clear message; never fabricate outputs.
2. Grad-CAM++ implemented in-repo (hooks on the last conv block of the first available fold model; no new dependency); heatmap overlay PNG per image; derived box from the thresholded heatmap (threshold from localization.json if present, else Otsu); localization_basis = "heatmap+derived_box". No box when the verdict is ACCEPT.
3. Output: <output>/predictions.csv (file, decision, family, confidence, flag, novel_score, heatmap_path, box_xyxy, top2, conformal_set) and predictions.parquet with the contract columns.
4. scripts/eval_localization.py: deletion/insertion AUC on a sample of validation units, and pointing-game / IoU against a classical-CV pseudo-GT mask (Otsu or difference from a median background; LABELED pseudo-GT, not human labels). Writes artifacts/localization.json with n, method and caveats.
5. scripts/build_demo_heatmaps.py: precompute heatmaps + boxes for all UNCERTAIN/REVIEW/NOVEL units, 20 per class, and crack_00011, rust_00187, rust_00891; update predictions.parquet in place, touching only heatmap_path, box_xyxy, localization_basis (keep a .bak).
6. Tiling for images larger than 256 px with tile_boxes and mixed_defect_flag when two families fire on different tiles: CUT LIST, do last.
7. Until real weights exist, test with a randomly initialised model saved as a temporary fp16 fold file in a temp dir. tests/test_infer.py checks shapes; a rotated, a noisy, a large and an RGB image; output columns; and that missing weights fail loudly. Never commit fake weights to artifacts/models/.
Report measured CPU latency per image, what ran, and what you could not verify.
```

### C9 - Pipeline, status, holdout guard, README reproduce section
```
OWNS: run_pipeline.py (extend C8's), scripts/eval_locked_holdout.py, scripts/write_status.py, tests/test_pipeline.py, tests/test_holdout_guard.py, the "Reproduce in one command" and assumptions sections of README.md, requirements.txt (pins only if a package is imported but missing).
1. run_pipeline.py --stage {ingest,vision,calib,rootcause,flow,profit,recs,infer,all}: each stage calls its existing script in dependency order (build_calibration, build_demo_heatmaps, build_sim_layer, merge_sim_into_predictions, run_rootcause then --recovery, build_flow, build_profit, build_recs, eval_localization); idempotent; skips a stage whose output is newer than its inputs unless --force; logs to artifacts/logs/<stage>.log; prints a per-stage table (ok | skipped | failed, seconds). --stage all uses kaggle_downloads/<run>/ (argument --kaggle-run) if present, else the artifacts already there, and ends with scripts/validate_artifacts.py.
2. scripts/write_status.py writes artifacts/status.json (component, status, evidence file, generated_at) derived ONLY from artifact existence and validation results. The Trust page's build-status table reads it.
3. scripts/eval_locked_holdout.py: the ONLY script allowed to read rows with split == 'holdout'. Requires --confirm, prints a warning banner, increments artifacts/holdout_runs.json, refuses when the counter is already >= 2 unless --override, uses the shipped ensemble through the SAME infer code path (no tuning, no threshold changes, no temperature refit), and writes artifacts/holdout_result.json: n, accuracy, macro-F1, per-class precision/recall, confusion matrix, FRR/FAR, expected cost, ECE, conformal coverage, UNCERTAIN/NOVEL/REVIEW rates. A --dry-run flag runs it on a fake tiny split for tests. YOU MUST NOT RUN IT ON THE REAL SPLIT; a human does that once.
4. tests: the guard refuses without --confirm and at counter >= 2; a grep test proves nothing else reads split == 'holdout'; run_pipeline --stage all works on fixtures in a temp copy.
5. README.md: a "Reproduce in one command" section (venv, requirements, run_pipeline.py --stage all, streamlit run app.py, infer usage) and the assumptions register (D14, D18, D22, D24, D25), only with what actually runs.
Report exactly what ran.
```

### C9b - Final README + model card (at the freeze, 06:30)
```
OWNS: README.md, MODEL_CARD.md. Read artifacts/*.json|csv only for numbers; never type a metric. Write the model card (intended use, data, training recipe, validation protocol, calibration, robustness table, novelty, localization with pseudo-GT caveat, holdout result if artifacts/holdout_result.json exists, limitations, assumptions register D14/D18/D22/D24/D25, ADVISORY/SIMULATED statement) and finalize README sections 1-8 using the rubric words as headings. Every number cites its artifact file. State plainly that image labels and model outputs are real while batch/process data are a labeled simulation because the organizers' files contain none. Report which numbers came from which artifact and anything missing.
```

---

## 6. Antigravity cards

Prepend the SHARED RULES block. **AG-2 (Inspect + Upload + Command Center) was issued separately; run A3 next.**

### A3 - Flow, Root Cause, Profit, Actions, Trust pages
```
OWNS: app/, app_next.py or app.py (per the current state), .streamlit/, tests/test_ui_smoke.py.
Work against tests/fixtures/ first (C1 creates them) and switch to real artifacts as they land; every page shows a graceful "artifact missing: run <command>" panel. All numbers are read from artifacts, nothing typed. Every number shows unit and source (caption or tooltip). ADVISORY/SIMULATED badge on all pages; assumption ids visible via assumptions_expander.
- Flow & Bottleneck: model selector (1, 2, 3 snapshot); utilization strip with bottleneck marker; THREE evidence lines (utilization, queue/WIP, starvation) with numbers from flow_metrics.json; twin-validation panel (MAPE, pass/fail from twin_validation.json); rework slider using rho_new = rho_baseline x (1 + rework_rate), labeled ASSUMPTION, with the saturation point.
- Root Cause: drift timeline per family with change-point markers (batch_metrics); evidence cards (tier, effect size, p_adj, lag, caveat, verify action); a "validation by planted-effect recovery" panel reading rootcause_recovery.json (planted, found, false positives, no-link control) that states the layer is simulated (D24) and that the engine never reads the truth file.
- Profit & What-if: sliders (decision threshold, defect reduction per family, bottleneck capacity, rework rate) over sim_results.parquet -> throughput and margin with 5-95% interval; waterfall; costs shown as ASSUMED with source config.yaml economics.
- Actions: recommendations.json ranked, with expected delta and interval, evidence chips linking to Root Cause cards, tier, invalidators, verify action.
- Trust: calibration (reliability, ECE raw vs scaled, temperature and whether a bound was hit); conformal coverage and set sizes; risk-coverage; robustness table from robustness_orient.csv sorted worst first with a flag below 0.95; LOFO novelty from novelty.json (per-family AUROC, "N of 4 families run"); localization from localization.json labeled "pseudo-GT"; holdout_result.json when present ("not yet run" otherwise); assumptions register; build status from status.json.
Add AppTest smoke tests per page on fixtures and on real artifacts. Done when pytest -q passes and every page renders with real or fixture data. Screenshots at 1440x900 under docs/screens/ (not committed). Report what you could not verify.
```

### A4 - Cross-links + Demo Mode
```
OWNS: app/, app_next.py or app.py, docs/demo_script.md, tests/test_ui_smoke.py.
1. Cross-links: unit (Inspect) -> its batch (Root Cause, highlighted) -> its evidence card -> its station (Flow) -> dollars (Profit what-if with the family preselected). Use st.session_state plus st.switch_page or query params; each hop shows a breadcrumb chip and a back link.
2. Demo Mode (sidebar toggle): a 3-minute guided walkthrough with Next/Prev and a timer: Command Center drift alert -> Inspect crack_00011 (heatmap, box, confidence, costs) -> Upload (judge picks an image) -> an UNCERTAIN/NOVEL example + LOFO result -> Root Cause with the recovery panel -> Flow with three evidence lines and twin validation -> Profit slider + top recommendation -> Trust page + holdout number if present. Each step pins the right page/unit and shows a one-line speaker cue read from docs/demo_script.md (create it from the demo section of docs/NEURAX_GodTier_Plan_v2.md).
3. The speaker cues must say: image labels and model outputs are real; batch and process data are a labeled simulation because the organizer files contain no such fields; the root-cause engine is validated by recovering planted effects and would run unchanged on real batch data.
4. Offline check: block the network and confirm no page requests an external URL (grep the code and check the browser network tab).
Commit explicit paths. Report.
```

### A5 - Visual and functional QA loop
```
OWNS: app/ and tests/ (fix defects you find in your own files only; report defects in other agents' files).
Use your browser tool on http://localhost:8501 at 1440x900 and 1280x720. For each of the 7 pages plus Demo Mode: screenshot and list defects: main insight not visible without scrolling, default light-blue Plotly colors, missing units or sources, numbers not traceable to an artifact (grep app/ for numeric literals that look like metrics such as 0.99, 97., 0.397 and report each), console errors, cold-load time, missing ADVISORY badge, truncated text. Fix what is yours, re-screenshot, repeat until a pass finds nothing. Then run the offline test: block the network, reload every page, confirm nothing fails. Also confirm streamlit run app.py works from a fresh terminal with only .venv. Report a defect table (found/fixed/left) and a pass/fail per page.
```

### A6 - Ingest Kaggle outputs, first full pipeline run
```
OWNS: artifacts/models/, and commits of generated artifacts under artifacts/ (json/csv/parquet <= 5 MB each) plus tests. Do not edit other agents' scripts.
The user downloaded the Kaggle output zip into kaggle_downloads\<run>\ (the folder name is given in the message).
1. List the files, compare with the expected per-fold outputs (read notebooks/train_5fold.py for names), report missing files per fold and each file's size. If any fold is incomplete, proceed with the complete folds and say which.
2. Copy fold{k}.pt of complete folds into artifacts/models/ (each < 50 MB; report sizes). Never commit anything from kaggle_downloads/.
3. Run one at a time, showing the output: scripts/build_calibration.py, scripts/build_demo_heatmaps.py, scripts/build_sim_layer.py, scripts/merge_sim_into_predictions.py, scripts/run_rootcause.py then run_rootcause.py --recovery, scripts/build_flow.py, scripts/build_profit.py then build_recs.py, scripts/eval_localization.py, scripts/write_status.py, scripts/validate_artifacts.py, pytest -q. If a script errors, report the traceback and stop that branch; do not patch other agents' files.
4. Print verbatim, without changing rounding: temperature T and whether a bound was hit; ECE raw and scaled; conformal q_hat and coverage (clean and shifted); UNCERTAIN, REVIEW and NOVEL counts; the worst 5 rows of robustness_orient.csv; macro-F1 per fold; the planted-effect recovery counts; the bottleneck calls; twin MAPE; the top 3 recommendations.
5. Restart streamlit and screenshot Command Center, Inspect and Trust.
Report.
```

---

## 7. What I check before anything reaches the screen or your mouth

1. Temperature T: is it still stuck on a bound? If ECE is tiny only because of a bound hit, do not present it as a result.
2. The shifted-set numbers (where errors exist) matter more than the clean 1.000: clean accuracy is saturated on this benchmark and separates nobody.
3. Every stress row below 0.95 needs a decision before the Trust page shows it.
4. Any number in the UI must trace to an artifact. Any number in a prompt or in this file is a placeholder, not a result.
5. The planted-effect recovery counts come from the engine's own run, never from the truth file.
6. The holdout number is shown once, exactly as printed, with n and the split definition.
7. The one-sentence honesty line for the jury: image labels and model outputs are real; batch and process data are a labeled simulation because the organizer files contain no such fields.
