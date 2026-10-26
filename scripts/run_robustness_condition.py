"""
Generic dispatcher for one (task, model, probe, axis, severity) run --
what a Slurm array job's SLURM_ARRAY_TASK_ID indexes into via
generate_robustness_manifest.py's manifest. Also the runner for plain
(non-robustness) feature extraction + probing when axis='clean' -- there
is no separate "normal" pipeline vs. "robustness" pipeline, clean is just
another (degenerate, no-op) condition.

2026-09-17 rewrite: the original version assumed a uniform low-level
`model.encode_pc(pc)` / `lib.encode_pc(model, pc)` call existed across
all 6 libs (built against ULIP-2, which happens to expose it). It
doesn't -- PointCLIP V2, Uni3D, Point-Bind and JM3D only expose the
high-level `encode_pointclouds(model, samples, ...)` used everywhere
else in this codebase, and OpenShape's build_model() returns a 3-tuple
(model, clip_model, tokenizer), not a single model. Running the first
full array batch this way produced 1,470/1,764 silent failures (a
second bug: the .slurm templates never propagated the failing exit
code, so Slurm reported every task COMPLETED regardless). Fixed by
routing through each lib's own validated encode_pointclouds() -- for
corrupted axes, corrupted copies are written to a temp dir first (since
encode_pointclouds reads samples from disk paths, it can't take
in-memory arrays), then cleaned up after.

Usage:
  python run_robustness_condition.py --task T1_organ_id --model ulip2 \
      --probe clustering --axis clean --severity none
  python run_robustness_condition.py --manifest robustness_manifest.tsv --idx 42
"""

import argparse
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import task_definitions as td
import robustness_corrupt as rc
from clustering_probe import clustering_metrics

MODEL_LIB = {
    'ulip2': 'ulip_zeroshot_lib',
    'openshape': 'openshape_zeroshot_lib',
    'pointclipv2': 'pointclipv2_zeroshot_lib',
    'uni3d': 'uni3d_zeroshot_lib',
    'pointbind': 'pointbind_zeroshot_lib',
    'jm3d': 'jm3d_zeroshot_lib',
}

OUT_ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi/robustness_results')

# Encoded features are expensive to produce (model-dependent, but PointCLIP
# V2's multi-view rendering alone took ~45min for 14k shapes across 7
# tasks) and were previously discarded after each run -- only the scalar
# metric got saved, so any later reuse (different prompts, a new probe,
# etc.) meant re-encoding from scratch. Cache them for axis='clean' only
# (the reusable case -- corrupted-condition features are one-off noisy
# copies of the same underlying shapes, not worth the extra disk).
FEATURE_CACHE_ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi/feature_cache')


def _cache_clean_features(task_name, model_key, feats, gt, paths):
    FEATURE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_CACHE_ROOT / f'{task_name}__{model_key}__clean.npz'
    if out_path.exists():
        return
    # np.savez() silently appends '.npz' to any path that doesn't already
    # end with it -- a tmp name like '*.npz.tmp' actually lands as
    # '*.npz.tmp.npz', breaking the rename below. Keep the tmp name ending
    # in '.npz' so savez writes exactly where we tell it to.
    tmp_path = out_path.with_name(out_path.stem + '.tmp.npz')
    np.savez(tmp_path, feats=feats, gt=gt, paths=np.array(paths, dtype=object))
    tmp_path.rename(out_path)


def get_task(task_name):
    if task_name in td.ALL_TASKS:
        return td.ALL_TASKS[task_name]
    raise KeyError(f'Unknown task {task_name!r}. Known: {list(td.ALL_TASKS)}')


def _write_corrupted_samples(samples, axis, severity, tmp_dir):
    """Corrupt each (path, label) sample per robustness_corrupt.py's
    deterministic per-file seeding and write it to tmp_dir, returning new
    (corrupted_path, label) pairs. Needed because encode_pointclouds reads
    from disk paths, not in-memory arrays."""
    out = []
    for npy_path, label in samples:
        pc = np.load(npy_path).astype(np.float32)
        seed = rc._seed_for(npy_path, axis, severity)
        pc = rc.CORRUPT_FN[axis](pc, severity, seed)
        # partiality/downsampling change point COUNT -- models expect a
        # fixed-size input, so pad by repeating points back up to 10000
        # (repetition, not upsampling new geometry).
        if len(pc) < 10000:
            pad_idx = np.random.RandomState(seed).choice(len(pc), 10000 - len(pc))
            pc = np.concatenate([pc, pc[pad_idx]], axis=0)
        out_path = tmp_dir / npy_path.name
        np.save(out_path, pc)
        out.append((out_path, label))
    return out


def run_one(task_name, model_key, probe, axis, severity, device='cuda', batch_size=32):
    task = get_task(task_name)
    lib = importlib.import_module(MODEL_LIB[model_key])

    samples = td.collect_samples_multiroot(task['folder_label_root'])
    if not samples:
        print(f'SKIP: no samples for {task_name} (data not downloaded yet?)')
        return None

    # OpenShape's build_model() returns (model, clip_model, tokenizer) and
    # needs clip_model+tokenizer for text encoding, not the point model --
    # every other lib returns a single model object for both calls.
    built = lib.build_model(device=device)
    if model_key == 'openshape':
        model, clip_model, tokenizer = built
        text_features = lib.encode_class_text(clip_model, tokenizer, task['class_names'], device=device)
    else:
        model = built
        text_features = lib.encode_class_text(model, task['class_names'], device=device)

    tmp_dir = None
    if axis != 'clean':
        tmp_dir = Path(tempfile.mkdtemp(prefix=f'corrupt_{task_name}_{model_key}_'))
        samples = _write_corrupted_samples(samples, axis, severity, tmp_dir)

    try:
        feats, gt, _paths = lib.encode_pointclouds(model, samples, batch_size=batch_size,
                                                     num_workers=8, device=device)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    feats = feats.cpu().numpy() if hasattr(feats, 'cpu') else np.asarray(feats)
    gt = gt.cpu().numpy() if hasattr(gt, 'cpu') else np.asarray(gt)

    if axis == 'clean':
        _cache_clean_features(task_name, model_key, feats, gt, _paths)
    text_features_np = (text_features.cpu().numpy() if hasattr(text_features, 'cpu')
                         else np.asarray(text_features))

    logits = feats @ text_features_np.T
    zs_preds = logits.argmax(axis=1)

    if probe == 'zeroshot':
        from sklearn.metrics import balanced_accuracy_score
        result = dict(bal_acc=balanced_accuracy_score(gt, zs_preds))
    elif probe == 'knn':
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import balanced_accuracy_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = []
        for train_idx, test_idx in skf.split(feats, gt):
            scaler = StandardScaler().fit(feats[train_idx])
            clf = KNeighborsClassifier(n_neighbors=5).fit(
                scaler.transform(feats[train_idx]), gt[train_idx])
            preds = clf.predict(scaler.transform(feats[test_idx]))
            accs.append(balanced_accuracy_score(gt[test_idx], preds))
        result = dict(bal_acc=float(np.mean(accs)), bal_acc_std=float(np.std(accs)))
    elif probe == 'clustering':
        cm = clustering_metrics(feats, gt)
        result = dict(bal_acc=cm['clust_bal_acc'], ari=cm['clust_ari'], nmi=cm['clust_nmi'])
    else:
        raise ValueError(f'Unknown probe {probe!r}')

    result.update(task=task_name, model=model_key, probe=probe, axis=axis,
                   severity=severity, n_samples=len(samples))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task')
    ap.add_argument('--model', choices=list(MODEL_LIB))
    ap.add_argument('--probe', choices=['zeroshot', 'knn', 'clustering'])
    ap.add_argument('--axis', default='clean')
    ap.add_argument('--severity', default='none')
    ap.add_argument('--manifest', help='TSV from generate_robustness_manifest.py')
    ap.add_argument('--idx', type=int, help='row index into --manifest (or $SLURM_ARRAY_TASK_ID)')
    ap.add_argument('--batch-size', type=int, default=32,
                     help='Lower this on smaller-VRAM GPUs (e.g. a16, '
                          '~15GB) -- PointBERT-style encoders (ULIP-2, '
                          'OpenShape, JM3D) OOM at the default 32 on '
                          '10k-point clouds due to their O(N^2) '
                          'per-batch kNN grouping op.')
    args = ap.parse_args()

    if args.manifest:
        import csv
        with open(args.manifest) as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        row = rows[args.idx]
        task_name, model_key, probe, axis, severity = (
            row['task'], row['model'], row['probe'], row['axis'], row['severity'])
    else:
        task_name, model_key, probe, axis, severity = (
            args.task, args.model, args.probe, args.axis, args.severity)

    if axis != 'clean':
        severity = float(severity) if '.' in severity else int(severity)

    print(f'=== {task_name} | {model_key} | {probe} | {axis}={severity} ===')
    result = run_one(task_name, model_key, probe, axis, severity,
                      batch_size=args.batch_size)
    if result is None:
        return

    print(result)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_name = f'{task_name}__{model_key}__{probe}__{axis}_{severity}.npy'
    np.save(OUT_ROOT / out_name, result, allow_pickle=True)
    print(f'Saved {OUT_ROOT / out_name}')


if __name__ == '__main__':
    main()
