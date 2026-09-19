# NEURAX 3.0 · QualityTwin — GOD-TIER PLAN v2

Supersedes §9 (timeline) and §10 (cut list) of `NEURAX_Master_Plan.md`. Everything else there still stands: §5A (real data facts), §6 (artifact contract), §11 (mistakes ledger), §12 (decisions). New decisions are D21+ at the end.

**Snapshot:** 2026-09-20 · team of 3 humans + coding agents (Codex, Antigravity).
**FILL IN NOW:** hard submission deadline = `________` · hours left = `____` · CP2 already passed? `Y / N`.
If less than ~5 h remain, jump to §11 (cut list) before starting anything.

---

## 0. The 60-second version

Re-reading the problem statement against what we have built, five things change the plan:

1. **CP3 is 60 marks and every vision mark is scored on the jury's unseen samples** (accuracy 15, localization 10, robustness 10, FR/FA 5). Robustness names *lighting, orientation, batch and novel variants*. **Our logged stress results cover noise, blur, JPEG, low-res, motion blur and speckle. Rotation, flips, scale, crop and perspective are not in the results we have.** That is the biggest hole between "0.9997 on our fold" and "works on any image".
2. **The jury said the model must detect crack / hole / rust / scratch on *any* image.** So the shipped system needs an `infer` path (any folder, any size, any format) and an **Upload page** in the UI, not just a dashboard over a pre-scored parquet.
3. **Localization is scored on unseen samples as "boxes, masks or heatmaps".** We must emit a heatmap *and* a derived box per image at inference time, and quantify quality somehow without ground truth.
4. **"Unified" is the whole point.** The statement literally says *not* another classifier / not an isolated dashboard. Root cause, twin, profit and recommendations are missing, and they are what turn our vision work into the system the jury asked for.
5. **Root cause (5 marks) has no real data behind it.** The only honest way to win those marks is a *labeled-simulated* factory layer plus a **planted-effect recovery test** that proves the engine works. The engine then runs unchanged on real batch data.

**The eight moves, in priority order:**

| # | Move | Marks it feeds |
|---|---|---|
| 1 | Close the "any image" gap: orientation/scale/perspective stress test, retrain 5-fold with full augmentation, ensemble + TTA | Accuracy 15, Robustness 10, FR/FA 5 |
| 2 | `infer` stage + **Upload page**: any image → verdict, family, heatmap + box, calibrated confidence, NOVEL/UNCERTAIN | All vision marks, Technical 5, UI 5, and the jury's explicit ask |
| 3 | Novelty for **all four** defect families + fused flag | Robustness, Explainability 5 |
| 4 | Localization upgrade: derived boxes + pseudo-mask validation | Localization 10 |
| 5 | Simulated factory layer + root-cause engine validated by planted-effect recovery | Root cause 5, CP2 relatability |
| 6 | Digital twin validated on Model 1/2 + profit model + what-if + recommendations | "Unified" claim, CP2/CP3 demo |
| 7 | Operator-console UI + one-click **Demo Mode** | UI 5, plus how every other mark is perceived |
| 8 | `run_pipeline.py`, tests, README, model card, holdout evaluated **once** | Technical 5, credibility |

---

## 1. Honest forecast against the hidden set

I cannot promise a number for data I haven't seen. This is what the evidence supports.

| Scenario | What the hidden set looks like | What to expect | Evidence / risk |
|---|---|---|---|
| **A. Same source, same lighting range** (most likely: organizers usually hold out a slice of the same dataset) | Same generator, same 5 classes, single label per image | **Macro-F1 ≈ 0.99–1.00** | Val 1.000, `cond_holdout` 0.999, 3,549/3,550 correct in the parquet, no leak found |
| **B. Same source + realistic shifts** | Noise, blur, JPEG, low-res, brightness, **rotation, scale, crop** | ≥ 0.98 on the corruptions we tested. **Rotation/scale/perspective unmeasured.** | Step V1 measures this in ~10 min. Retrain with orientation augmentation closes it |
| **C. Different domain** | Real photos, other textures, other resolution, different defect look | **Unknown; could drop a lot.** | Our images *look* procedurally rendered (clean grey textures, drawn polylines, dark discs, blob outlines). A model can learn generator-specific cues. Mitigation: heavy augmentation, ensemble, novelty flag so it abstains instead of guessing; probe V7 tells us early |
| **D. Several defects in one image, or large images** | Multi-label / different aspect ratio | Single-label model returns one class | Mitigation: top-2 reporting, "multi-defect suspicion" flag, tiled inference for large images (V6) |

**What decides rank** (controllable): unseen-sample performance (40 of 60 CP3 marks), the unified story actually working end to end, a demo that never breaks, and honest caveats (judges reward stated limitations under "Explainability"). "Other teams are ahead" is about CP1/CP2 optics; CP3 is 60 marks measured on the final system.

---

## 2. Requirement traceability (problem statement → deliverable)

| Problem-statement phrase | Deliverable | UI page | Status |
|---|---|---|---|
| classify units as acceptable or defective | binary verdict + 4 families; cost-based ACCEPT/REJECT/REVIEW | Inspect | built (fold 0) → V2/V3 |
| explain what is wrong with a product | family + heatmap + class probs + nearest known examples | Inspect | partial |
| where the defect sits | Grad-CAM++ heatmap + derived box (+ pseudo-mask score) | Inspect overlay | notebook only → V5 |
| flag uncertain or novel defect types | fused UNCERTAIN / NOVEL flag at inference | Inspect, Upload | scratch only → V4 |
| what process condition is likely responsible | evidence cards (simulated linkage, validated by planted-effect recovery) | Root Cause | not started → P3/P4 |
| where the production flow is constrained | bottleneck with 3 evidence lines (utilization, queue, starvation) | Flow | v0 → P1 |
| how the constraint affects throughput and cost | twin + loss waterfall + value of an hour at the bottleneck | Flow / Profit | not started → P2 |
| how quality or process changes affect profitability | what-if grid (threshold, defect reduction, capacity, changeover) | Profit & What-if | not started → P5 |
| continuously updated view | replayed unit stream, play/pause/speed, live KPIs | Command Center | not started → U3 |
| evidence-based process recommendations | ranked ADVISORY actions with ΔTH, Δmargin (interval), evidence ids | Actions | not started → P6 |
| all simulated or advisory | global badge + assumption id on every derived number | everywhere | built |
| mixed variants · changing conditions · recurring families · batch drift · varying capacity | variant mix + drift in the simulated layer; lighting/orientation in stress suite; family-level root cause; capacity in twin | Root Cause / Flow | partial |

---

## 3. Architecture v2

```
 images ─► grouped splits ─► 5-fold EfficientNet-B0 (Kaggle) ─► OOF probs, embeddings, weights
                                     │
              ┌──────────────────────┼─────────────────────────────┐
              ▼                      ▼                             ▼
        calibration +          kNN novelty bank            Grad-CAM++ → heatmap + box
        conformal + cost       (4 families LOFO)           (+ pseudo-mask validation)
              └──────────────┬───────┴─────────────────────────────┘
                             ▼
                  infer (CPU, any folder)  ──►  predictions.parquet / predictions.csv
                             │
 production CSVs ─► flow + twin (validated on Model 1/2) ─► flow_metrics.json, sim_results.parquet
                             │
 simulated factory layer (batches, process params, planted effects; seeded, labeled)
                             ▼
        root-cause engine ─► evidence_cards.json      profit model ─► profit_model.json
                             └──────────────► recommendations.json
                                                   ▼
                       Streamlit operator console (reads artifacts; Upload page runs CPU inference)
```

**Principle update (D21):** the demo needs no internet and no GPU. **CPU inference is allowed** (Upload page), with precomputed artifacts as the fallback so a crash can't kill the demo.

---

## 4. Vision plan v2 (≈45% of effort; Kaggle, human-run, agent-written scripts)

**V1: Orientation/scale stress test (10 min, do first).** On the current fold-0 augmented model: rot 90/180/270 and arbitrary angles, h/v flips, scale 0.5×–2×, center crop 70%, perspective warp, gamma, shadow gradient, color temperature. Output `robustness_orient.csv`. *Decision rule:* any test < 0.95 → V2 must include that transform in training.

**V2: Retrain 5 folds with full augmentation.** Same model (EfficientNet-B0, D11) so results stay comparable. Add: rot90 + arbitrary rotation, flips, random-resized-crop (scale 0.6–1), perspective, brightness/gamma/contrast/shadow/color-temperature, plus the existing noise/blur/JPEG/low-res, plus cutout. Save per fold: weights (fp16), OOF probabilities, 1280-d embeddings. Checkpoint after **every fold** to `/kaggle/working` and run as "Save & Run All (commit)" so a disconnect cannot wipe it (Mistakes #9, and the disconnect you already had). Verify per-fold runtime from fold 0's log before committing to 5.

**V3: Ensemble + TTA + calibration.** Average the 5 fold models; TTA = flips + rot90. Temperature-scale on OOF (the previous fit hit its search bound because the model was near-perfect; with harder augmentation it should land in a sane range; widen the search bound anyway).
*FR/FA note:* on clean data the error counts are ~0, so the cost-vs-threshold curve is degenerate. **Report FRR/FAR and the cost curve on the corrupted/shifted validation set**, where errors exist and the trade-off is visible. Reconcile costs in one place (`config.yaml: economics`) and regenerate decisions (D22).

**V4: Novelty for all four defect families.** Leave-one-family-out for crack, hole, rust, scratch (scratch is done: kNN AUROC 0.991). Ship one kNN bank of training embeddings per class + threshold at 5% FPR on dev. **Fused flag:**
- `NOVEL` = embedding distance above threshold (far from every class)
- `UNCERTAIN` = conformal set empty/ambiguous, or top-2 probabilities close, or ensemble members disagree
- otherwise `NONE`
Report AUROC for all four hidden families in the Trust page.

**V5: Localization.** Grad-CAM++ for every unit (predicted class). Derive a box from the thresholded heatmap (largest connected components above 50% of max; `localization_basis = "heatmap-derived box"`). **Quality without ground truth:** build pseudo-masks with classical CV on a ~300-image sample (background subtraction + Otsu + morphology), then report pointing-game hit rate and IoU of heatmap-box vs pseudo-mask, **labeled pseudo-GT**. Keep deletion/insertion AUC (0.397 / 0.826). *Stretch:* train a small U-Net on the pseudo-masks for sharper overlays; kill it if rust pseudo-masks look bad.

**V6: Any size, any aspect, multi-defect.** If the image is not ~square or is much larger than 256 px: overlapping tiles (e.g. 256 px, 50% overlap), aggregate per-tile predictions, return tile boxes; flag `MULTI_DEFECT_SUSPECTED` when two confident tiles disagree on family.

**V7 (optional, 30 min cap): domain-gap probe.** 20–50 real defect photos (public Kaggle datasets can be added to a notebook without internet) run through the ensemble. **Probe only, not training**, because we don't know whether outside data is allowed (D18). The point is to learn early whether Scenario C is a disaster, and to see whether the NOVEL flag fires on it.

**V8: Final evaluation.** Ship the **5-fold ensemble** (the model that was evaluated is the model that ships). Evaluate on the **locked holdout (1,457) exactly once**, after feature freeze, by a human. No agent runs it.

---

## 5. Process, economics and the "simulated factory" (≈30%)

**P1: Flow, ground it in all three files.** Keep Model 1/2. Parse Model 3 (605,620 × 78, ~20 stations, single snapshot) into a per-station table and triangulate the bottleneck on utilization, queue/WIP and starvation. Model 2's Drilling-vs-Assembly conflict stays disclosed.

**P2: Digital twin validated against real data.** Build a small SimPy (or analytical + Monte Carlo) 3-station line for Model 1/2 and **check that its utilization-vs-Demand curves match the Arena results** (state the tolerance *before* looking, e.g. MAPE ≤ 10%; report the real number). Only then run what-ifs. This is the one place a validation against real production data is possible; it earns credibility for everything simulated.

**P3: Simulated factory layer (resolves D17: labeled-simulated).** `src/sim/make_batches.py`, seeded and documented. Take the model's out-of-fold predictions on real images, assemble a **replay stream** of ~40 batches × ~200 units with timestamps, variant (A/B), shift, supplier lot, and process parameters with drift and change-points. Defect families are sampled to follow **planted causal effects**:

| Family | Planted cause (assumption id A-cause) | Design |
|---|---|---|
| rust | coolant humidity above a threshold | 1-batch lag, change-point mid-run |
| scratch | fixture wear / changeovers per shift | gradual drift |
| hole | drill tool-wear hours, resets at tool change | sawtooth |
| crack | **no true cause planted** | engine must say "no link found" |
| nuisance | 4 pure-noise parameters + shift/supplier confounders | negative controls |

Bonus: the replay stream has a realistic ~3–12% defect rate, which fixes the "80% reject is a benchmark artifact" caveat *for the flow story*, clearly labeled.
Family→station **assumption map (A-map):** hole→Drilling, scratch→Milling, crack→Assembly, rust→environment. It lets the flow coupling become station-specific: ρ_s,new = ρ_s × (1 + rework_s).

**P4: Root-cause engine** (per master plan M3): p-charts/CUSUM, change-point (`ruptures`), stratified tests within variant, effect sizes, BH-FDR, LightGBM + SHAP with batch-grouped CV, shuffled-label baseline, evidence cards. **The headline artifact is the planted-effect recovery table:** which planted links were found (sign, lag, effect size), false links among nulls, and the deliberate "no link" for crack. Target: ≥ 3 of 3 planted links recovered at FDR 0.05, ≤ 1 false link among 4 nulls. Wording everywhere: *association, not proven causation; simulated linkage*.

**P5: Profit model.** Accounting identity from `config.yaml: economics` (price, variable cost, scrap, rework, escape/warranty, review, downtime per hour). What-if grid from the twin (threshold, defect reduction, capacity, changeover, variant mix) with intervals from ≥200 replications. ML cross-check on the simulated batches with a **time split**, labeled simulated. All economics are assumptions (D18), editable in the UI.

**P6: Recommendations.** Ranked ADVISORY actions: score = expected Δmargin (5–95% interval) × evidence tier; each shows why, impact, confidence, what could invalidate it, how to verify. LLM narrator optional and *narration only* (it never computes numbers), with template fallback.

---

## 6. UI/UX spec: the "operator console"

Stay on **Streamlit + Plotly** (D1). A Next.js rewrite is not worth the risk in this time. Polish it hard instead.

**Design system:** dark industrial theme via `.streamlit/config.toml` + one `theme.py` (palette, spacing, a Plotly template). Status pills: green ACCEPT, red REJECT, amber REVIEW/UNCERTAIN, violet NOVEL. Permanent **ADVISORY / SIMULATED** badge. Every number shows unit and source on hover/caption. KPI strip on every page. No walls of caption text: move caveats into "Assumptions & limits" expanders.

**Pages** (`st.navigation`):
1. **Command Center**: live replay (play/pause/speed via `st.fragment(run_every=…)`), FPY, defect rate by family, OEE-style flow KPI, margin/h, drift alerts. Clicking a drift alert jumps to Root Cause.
2. **Inspect + Upload**: unit browser *and* an **Upload box**: drop any image → verdict, family, calibrated confidence, conformal set, NOVEL/UNCERTAIN with the reason, heatmap overlay with opacity slider, derived box, nearest known examples, expected-cost table. "Send to Root Cause" / "Show station" buttons.
3. **Root Cause**: batch drift timeline, evidence cards (effect size, p_adj, lag, stability, tier, caveat, how to verify), planted-recovery panel.
4. **Flow & Bottleneck**: station utilization strip, WIP/queue bars, three-evidence bottleneck marker, loss waterfall, value of one hour at the bottleneck.
5. **Profit & What-if**: sliders (threshold, defect reduction, capacity, changeover, variant mix) → margin with interval; economics editor.
6. **Actions**: ranked recommendations with evidence links back to pages 3–5.
7. **Trust**: calibration/reliability, risk–coverage, FR/FA and cost curve on shifted data, robustness table (incl. orientation), novelty AUROC × 4 families, localization (faithfulness + pseudo-GT), assumptions list, build status.

**Cross-links** via `st.session_state`/`st.query_params`: defect → batch → cause → station → dollars. This is what stops it being "an isolated KPI dashboard".

**Demo Mode (button):** a scripted 2.5-minute guided walkthrough that steps through the story beats (§9) with auto-navigation and highlights. It doubles as the live-demo insurance.

---

## 7. Automation: how to use Codex and Antigravity without losing control

**Split by strength (if you only have one agent, run the cards in the listed order):**
- **Codex**: backend, tests, contracts, sim, root cause, flow, profit, pipeline. Well-specified, testable in isolation, parallel-safe (one branch each).
- **Antigravity**: UI, and **visual QA with its browser agent** (screenshot every page, critique against the checklist in T11). Also good for Kaggle-script editing in the IDE.
- **You (humans)**: run Kaggle jobs (agents can't), merge PRs, run the holdout once, own the demo.
- **Me (this chat)**: review diffs/output you paste, write or fix task prompts, debug failures, sanity-check numbers before they go on screen.

**Rules for every agent** (put in `AGENTS.md`; Codex reads it, paste the same text into Antigravity's workspace rules):

```
# AGENTS.md — QualityTwin
- Read NEURAX_Master_Plan.md (§5A data facts, §6 artifact contract) and NEURAX_GodTier_Plan_v2.md first.
- Windows + PowerShell, Python in .venv. App and inference must run CPU-only, offline.
- Contract-first: files in artifacts/ must validate against contracts/*.schema.json.
  Never rename or drop a field; add optional fields only.
- NEVER read, evaluate on, or reference the locked-holdout rows in artifacts/splits.csv.
  Only a human runs that evaluation, once.
- Never hard-code a metric into the UI or docs. Read it from artifacts/*.json; show "n/a" if missing.
- Anything derived from an assumption carries its assumption id (D14, D17, D18, A-cause, A-map, economics)
  and the ADVISORY/SIMULATED tag.
- Do not commit data/, kaggle_*/, .venv, or weights > 50 MB.
- One task = one branch (agent/<task-id>). Small commits. `pytest -q` must pass before you say done.
- Final report = files changed, how to run it, what you could NOT verify. Do not claim what you did not run.
- Do not edit files outside your task's allowed list.
```

**Guardrails that matter for a hackathon:**
- **Contract tests first (T1)**: agents that break the artifact contract break the demo. Every merge runs `scripts/validate_artifacts.py`.
- **One human merges**, always into `main` only from a green branch. Keep a tag `cp2-demo` on today's working state so there is always something to show.
- **Agents invent numbers.** Rule above plus: I review any metric before it reaches the UI.
- **Kaggle is the bottleneck.** Agents write scripts that run top-to-bottom, save after every fold, and print a final "DONE + file list". You run them.
- **Data exposure:** cloud agents clone the repo, and the repo currently contains the 12,000 organizer images (D15 said private). Before pointing a *cloud* agent at it: confirm the repo is private, and at minimum `git rm -r --cached data kaggle_upload` + `.gitignore` so new clones don't carry a working copy. History cleanup can wait until after the event. Local agents avoid the issue.

---

## 8. Task cards (paste one per agent session)

Each card: **Goal · Inputs · Outputs · Allowed files · Done when**. Paste `AGENTS.md` first.

**S0: Repo hygiene (human, 15 min).** Tag current state `cp2-demo`. Check repo is private. `git rm -r --cached data kaggle_upload`, add both to `.gitignore`, commit. Create `contracts/`, `AGENTS.md`, branches. Launch T1, T5, T7, T9 in parallel.

**T1 (Codex): contracts + validators.**
Goal: JSON Schemas for every artifact in master plan §6 (start from the real `predictions.parquet` columns: unit_id, image_path, batch_id, variant, condition, station, ts, p_defect, class_probs, pred_family, decision, flag, conformal_set, anomaly_score, heatmap_path, boxes_mask_ref, localization_basis, true_label, source_split) plus `evidence_cards`, `flow_metrics`, `sim_results`, `profit_model`, `calibration`, `robustness`, `recommendations`. Add optional fields: `box_xyxy`, `novel_score`, `top2`, `tile_boxes`, `mixed_defect_flag`.
Outputs: `contracts/*.schema.json`, `scripts/validate_artifacts.py`, `tests/test_contracts.py`, tiny fixture artifacts in `tests/fixtures/`.
Allowed: `contracts/`, `scripts/validate_artifacts.py`, `tests/`.
Done when: the current `artifacts/predictions.parquet` and `flow_v0.json` validate; a deliberately broken fixture fails with a clear message.

**T2 (Codex): `infer` stage.**
Goal: `python run_pipeline.py --stage infer --input <folder> --out <dir>` scores any image folder on CPU.
Behavior: loads the 5 fold weights (`artifacts/models/fold{k}.pt`, fp16) + kNN bank + temperature + conformal q̂ from `artifacts/calibration.json`; grayscale/RGB/any size/jpg-png-bmp; TTA (flips, rot90); per image writes verdict, family, top-2, calibrated confidence, conformal set, flag (NONE/UNCERTAIN/NOVEL) with reason, Grad-CAM++ heatmap PNG, derived box; tiled mode for large/non-square images; a corrupt file is logged and skipped, never crashes the run. Also expose `infer_one(path) -> dict` for the UI.
Outputs: `src/vision/infer.py`, `run_pipeline.py`, `predictions.csv/.parquet` + `heatmaps/`, tests.
Allowed: `src/vision/`, `run_pipeline.py`, `tests/`.
Done when: 100 images run on CPU in < 3 min; probabilities match the Kaggle notebook to 1e-3 on 20 reference images; output validates against T1 schemas. **Blocked until** T3 delivers weights; until then develop against the fold-0 weights.

**T3 (Codex writes, you run on Kaggle): 5-fold training script.**
Goal: `notebooks/train_5fold.py` implementing V2/V3. Same model and recipe as D11 plus the augmentation list in V2; groups from `splits.csv`; **never touches the locked holdout**. After each fold save weights (fp16), OOF probs, embeddings, and a JSON log; print `DONE fold k` plus file list. Then an ensemble script that computes OOF temperature, conformal q̂, cost table, reliability data, and writes `calibration.json`.
Done when: dry-run on 200 images per class passes locally on CPU; a human runs it on Kaggle in "Save & Run All" mode.

**T4 (Codex writes, you run): stress + novelty scripts.**
Goal: `notebooks/stress_orientation.py` (V1) and `notebooks/novelty_lofo.py` (V4, all four families, AUROC for softmax/energy/kNN, ship the bank and 5%-FPR threshold). Outputs `robustness_orient.csv`, `novelty.json`, `knn_bank.npz`.

**T5 (Codex): simulated factory layer (P3).**
Goal: `src/sim/make_batches.py` + `config.yaml: sim` block; seeded; input `predictions.parquet` (or OOF parquet), output `batch_metrics.parquet` and a time-ordered `replay_stream.parquet` (unit_id, ts, batch_id, variant, shift, supplier_lot, station attribution via A-map). Plant the effects in the P3 table exactly; write the *true* planted coefficients to `sim_truth.json` (used only by the recovery test, never shown as data). Document every assumption id.
Done when: same seed → identical output; defect rate per batch lands 3–12%; null parameters are statistically unrelated to family mix.

**T6 (Codex): root-cause engine (P4).**
Goal: `src/rootcause/` implementing the P4 steps, reading only `batch_metrics.parquet` (never `sim_truth.json`). Writes `evidence_cards.json` and `rootcause_recovery.json` comparing findings with `sim_truth.json` (recovered / missed / false links; crack must come back "no link").
Done when: recovers all planted links with correct sign at FDR 0.05, ≤ 1 false link among nulls, shuffled-label baseline shows no signal, and a unit test asserts the "no link" case.

**T7 (Codex): flow + twin (P1, P2).**
Goal: extend `src/flow/`: Model 3 station table and 3-evidence bottleneck; SimPy (or analytical + Monte Carlo) twin for the 3-station line, calibrated on Model 1/2; validation report (`twin_validation.json`: simulated vs observed utilization across Demand, MAPE); loss waterfall using editable assumptions; `flow_metrics.json`; `sim_results.parquet` for the what-if grid (≥ 200 replications, CIs).
Done when: validation MAPE is reported honestly (state the tolerance in the file *before* the run; do not tune to hit it silently); what-if grid regenerates from one command.

**T8 (Codex): profit + recommendations (P5, P6).**
Goal: `src/profit/`, `src/recs/`, `config.yaml: economics`. Accounting identity, time-split ML cross-check with intervals, ranked `recommendations.json` (each with ΔTH, Δmargin interval, evidence ids, tier, assumptions, "what could invalidate it", "how to verify"); template narrator with an optional LLM narrator that only rephrases the JSON and falls back to templates offline. Single source of truth for costs; regenerate ACCEPT/REJECT/REVIEW from it.

**T9 (Antigravity): UI shell + design system.**
Goal: split `app.py` into `app/` pages with `st.navigation`; `theme.py`, `.streamlit/config.toml`, shared components (KPI strip, status pill, evidence card, assumptions expander, ADVISORY badge); page stubs reading real artifacts, `n/a` when missing.
Allowed: `app/`, `.streamlit/`.

**T10 (Antigravity): pages.** Build the seven pages in §6 against the contract, one page per commit, in this order: Inspect+Upload → Command Center → Flow → Root Cause → Profit → Actions → Trust. Upload page calls `infer_one` (mock it until T2 lands). Add cross-links and Demo Mode last.

**T11 (Antigravity): visual QA loop.** Run the app; with the browser agent, screenshot every page at 1440×900 and 1920×1080 and check: no clipped text, no default-blue Plotly bars (use the theme), every number has a unit/source, ADVISORY badge visible, no page needs scrolling to reach its main insight, Upload works with a jpg, a png, a huge image, and a corrupt file, and nothing breaks with Wi-Fi off. Fix what it finds; list what it can't.

**T12 (Codex): pipeline, docs, repro.**
Goal: `run_pipeline.py --stage {ingest,vision,calib,rootcause,flow,profit,recs,infer,all}` (D20), pinned `requirements.txt`, smoke tests, `MODEL_CARD.md`, README consolidation (rubric headings), clean-environment test script. Done when `--stage all` runs from a fresh venv using cached weights.

---

## 9. Timeline (scale to the hours you actually have)

Assumes ~6 h. Multiply each block by (hours left ÷ 6); if you have less, apply §11.

| T-minus | Humans | Agents |
|---|---|---|
| 6:00–5:45 | S0. Fill in the deadline. Launch V1 stress test on Kaggle | T1, T5, T7, T9 start |
| 5:45–5:00 | Review T3/T4 scripts, launch 5-fold training (background, ~1–1.5 h; check fold-0 runtime first) | T6 starts once T5 output exists; T10 starts on Inspect+Upload |
| 5:00–3:30 | While Kaggle runs: review PRs, run P3→P4 locally, check twin validation, spot-check numbers | T2 (against fold-0 weights), T8, T10 continue |
| 3:30–3:00 | **Gate G1:** 5-fold weights + calibration in hand? If not, ship fold-0 + TTA and drop V3 ensemble | T2 switched to ensemble weights; T4 novelty + bank land |
| 3:00–2:00 | Integrate everything; `--stage all` from clean venv; V5 pseudo-GT numbers | T11 visual QA rounds; T12 |
| **2:00** | **FEATURE FREEZE (G2).** Tag `rc1` | Only bug fixes |
| 2:00–1:15 | **Locked holdout, once.** Model card, README, Trust page filled from artifacts | T12 finish |
| 1:15–0:30 | Rehearse the demo 3×, record backup video, test offline + second machine + Upload with judge-style images | T11 final sweep |
| 0:30–0:00 | Buffer. Tag `v1.0`. Submit | None |

---

## 10. Demo script v2 (3 min; Demo Mode automates it)

1. **0:00 Command Center:** one-sentence problem; live replay running; point at the drift alert.
2. **0:25 Inspect:** a defective unit → verdict, family, heatmap + box, confidence, expected-cost table.
3. **0:50 Upload (let a judge pick an image, ideally a rotated or noisy one):** live verdict + heatmap in seconds.
4. **1:20 Abstention:** a NOVEL/UNCERTAIN unit and why; leave-family-out AUROC across all four families.
5. **1:40 Root Cause:** the drift alert's evidence card; the recovery panel ("we planted 3 causes, found 3, and correctly found none for crack").
6. **2:00 Flow:** bottleneck with three evidence lines, waterfall, twin validation vs Arena data.
7. **2:20 Profit & Actions:** move the defect-reduction slider → margin with interval → top recommendation.
8. **2:45 Trust:** calibration, orientation/corruption table, limits. Close with the locked-holdout number.

**Say out loud:** "Image labels and model outputs are real; batch and process data are a labeled simulation because the organizers' files contain no batch or defect fields; the engine is validated by recovering planted effects and runs unchanged on real batch data."

---

## 11. Cut list and kill criteria

Cut in this order when time runs short:
1. Variant-mix slider and LLM narrator (templates only).
2. U-Net on pseudo-masks (keep derived boxes + pseudo-GT numbers).
3. Model 3 triangulation (keep Model 1/2).
4. Tiled inference and domain probe (keep the multi-defect flag off).
5. ML profit cross-check (keep the accounting identity).
6. Demo Mode (keep the manual script).

**Never cut:** V1/V2 orientation + augmentation, `infer` + Upload, the honest Trust page, `--stage all`, the holdout run, the backup video.

Kill criteria: 5-fold not finished by **T-3:00** → ship fold-0 + TTA. Twin validation MAPE > 25% → analytical model + Monte Carlo, say so. Planted-effect recovery fails → fix the engine, do not loosen the plant. Any agent output that breaks contract tests → revert, don't patch around.

---

## 12. Extra questions to prepare (adds to master plan §13)

- **"Does it work on a rotated / different-size image?"** Show the orientation table and the Upload page live.
- **"Your batch data is simulated. Why should we trust the root cause?"** We say so up front. The engine is validated by recovering effects we planted, it correctly reports no link where we planted none, and it runs unchanged on real batch data.
- **"What if an image has two defects?"** Top-2 + multi-defect flag + tiled mode; single-label training is a stated limitation.
- **"How do you know the box is right without masks?"** Pseudo-GT from classical CV, labeled as such, plus faithfulness curves; no IoU claim against organizer masks.
- **"Where do the economics come from?"** Editable assumptions in one config; the UI shows them and the sensitivity of margin to each.
- **"Why is the twin credible?"** Validated against the organizers' Arena utilization-vs-demand curves before any what-if.

---

## 13. New decisions (append to master plan §12)

| ID | Decision | Status |
|---|---|---|
| D21 | Demo needs no internet/GPU; CPU inference allowed (Upload page), precomputed artifacts as fallback | firm |
| D22 | One source of truth for costs (`config.yaml: economics`); ACCEPT/REJECT/REVIEW regenerated from it | provisional |
| D23 | Ship the 5-fold ensemble; evaluate it once on the locked holdout; no all-data retrain unless time remains and it is stated | provisional |
| D24 | D17 resolved: batch/variant/process/ts are **labeled-simulated**, seeded, with planted effects validated by a recovery test | provisional |
| D25 | A-map: hole→Drilling, scratch→Milling, crack→Assembly, rust→environment (assumption, editable) | provisional |
| D26 | Localization = heatmap + derived box; quality reported vs pseudo-GT (classical CV masks) and faithfulness | provisional |
| D27 | Public real-image datasets used as a probe only, never for training, until rules are known | provisional |
| D28 | Agents work on branches; a human merges; no agent touches the locked holdout | firm |
