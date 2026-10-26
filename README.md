# When Zero-Shot Fails but Features Don't

Code and results for *"When Zero-Shot Fails but Features Don't: Isolating the Text-Decision Bottleneck in 3D Shape-Language Models on Medical Anatomy"*.

We evaluate six 3D shape-language models (ULIP-2, OpenShape, PointCLIP V2, Uni3D, Point-Bind, JM3D) on a 277-subject MedShapeNet cohort with three probes on the same frozen embeddings: zero-shot (text prompts), supervised kNN, and unsupervised k-means (Hungarian-matched). We also evaluate a training-free few-shot cache-adapter.

## Layout

| Path | Contents |
|---|---|
| `cohort/cohort_sids.txt` | The 277 MedShapeNet subject IDs used |
| `cohort/tasks_categories.txt` | MedShapeNet categories assigned to each task (T1–T6) |
| `cohort/download_categories.txt` | Categories downloaded from MedShapeNet |
| `scripts/build_cohort277.py`, `task_definitions.py`, `robustness_corrupt.py` | Cohort download and preprocessing (10k points, unit-sphere), task definitions, point-cloud corruptions |
| `scripts/*_zeroshot_lib.py` | Wrappers for the six models |
| `scripts/populate_feature_cache.py`, `run_robustness_condition.py`, `clustering_probe.py` | Embedding extraction and the zero-shot, kNN and clustering probes (clean and corrupted) |
| `scripts/cache_adapter.py` | Few-shot cache-adapter |
| `scripts/soft_prompt_tuning.py`, `combined_fix.py`, `informed_cache_adapter.py`, `cluster_consensus.py` | Alternatives reported in the Limitations |
| `scripts/measure_inference_time.py`, `pointclipv2_prompt_ablation.py`, `check_pathology_confound.py` | Inference timing, prompt ablation, segmentation-confound check |
| `results/results_summary.csv` | All per-task results (every model × task × probe × corruption), including each of the 15 laterality structures separately |
| `results/main_grid_clean_aggregated.csv` | Clean-condition results aggregated per task group (Table 1) |
| `results/robustness_results/` | Raw per-condition robustness outputs |
| `results/cache_adapter_results/` | Few-shot cache-adapter, K = 1–16, 3 random draws |
| `results/soft_prompt_results/`, `combined_fix_results/`, `informed_cache_adapter_results/`, `cluster_consensus_results/` | Alternatives reported in the Limitations |
| `results/pointclipv2_prompt_ablation_results.csv` | PointCLIP V2 prompt ablation |

## Data and models

Shapes are not redistributed. Download them from [MedShapeNet](https://medshapenet.ikim.nrw/) under its license, then rebuild the cohort with `scripts/build_cohort277.py` and the subject-ID list. Model checkpoints come from each model's official release (see the paper, Sec. 2.3). Scripts contain absolute paths from our cluster that need adjusting.
