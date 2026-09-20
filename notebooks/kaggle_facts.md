# Kaggle reference facts (read-only inspection)

Inspected notebook: `notebook67fae4b0ae` at the existing Kaggle editor URL on 2026-09-20.

- Fold-0 training wall time: **not verifiable**. The visible notebook is a draft and every visible cell is marked “Cell hasn't been executed yet”; no fold-0 log was present.
- `/kaggle/working` `.pt` weights: **not verifiable** from the read-only editor. No output file listing was visible.
- Input dataset name and exact path: **not verifiable**. The reference cell searches recursively under `/kaggle/input` for `splits.csv`, then infers the image root from the first listed file. The code does not contain a dataset name.
- Whether `splits.csv` is inside the attached dataset and whether its assignments match `artifacts/splits.csv`: **not verifiable** because the reference cells were not executed and the dataset attachment was not exposed in the editor view.
- Accelerator: the Settings menu showed `None` selected and offered `GPU T4 x2`; no accelerator was enabled during inspection.
- Torch and timm versions: **not verifiable**. The environment-check cell prints them, but it was not executed.
- Remaining weekly GPU quota: **not visible** in the inspected notebook UI; therefore unverified.

The reference code visible in the editor uses `efficientnet_b0` (the `ARCH` value is referenced by the training cell), pretrained timm initialization, sqrt-inverse class weights, AdamW with `weight_decay=1e-2`, OneCycleLR with `pct_start=0.15`, CUDA AMP, fixed epochs with the last epoch retained, and 256px grayscale-to-3-channel inputs. The new trainer preserves that recipe and adds the requested resumable five-fold artifacts.

The local frozen split file was inspected only for schema/split counts needed to implement filtering: columns are `file, cls, mean_brightness, group, split, fold`; it contains `dev`, `cond_holdout`, and locked `holdout` partitions. Locked rows are excluded before any image load or scoring.

## New-notebook DRY_RUN verification

- Imported notebook title: `neurax-train-5fold`; visibility was left unchanged/private.
- Attached dataset: `dataset001` (`h3m4nth/dataset001`), verified in the run as `/kaggle/input/datasets/h3m4nth/dataset001`.
- Attached `splits.csv` was found. After the same locked-row filter, the embedded and attached metadata had zero differing rows; the comparison itself intentionally excluded the locked partition.
- Accelerator: `GPU T4 x2`; Kaggle reported two Tesla T4 devices.
- Kaggle versions: Python 3.12.13, torch 2.10.0+cu128, timm 1.0.26.
- Imported source hash matched the local trainer: `26e6f724f54c5df9419cd8e15d4b3f796a1592804988febbbe8c16a2c2522524`.
- DRY_RUN fold wall times from `log_fold{k}.json`: fold 0 `27.901198235000038` seconds; fold 1 `2.195922803999963` seconds.
- Printed GPU training throughput: fold 0 `7.17 img/s`; fold 1 `309.13 img/s`.
- Every expected output for folds 0 and 1 existed: weights, OOF/conditional/shift parquet files, three embedding banks, stress CSVs, and JSON logs.
- Weekly GPU quota remains unverified; it was not visible in the inspected UI.
- The full 12-epoch run was not started.
