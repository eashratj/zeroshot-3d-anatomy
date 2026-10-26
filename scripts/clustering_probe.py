"""
Generalized unsupervised clustering probe: k-means (k = #classes, 10 inits)
on a model's cached frozen embedding, clusters mapped to ground-truth
classes *after* fitting via Hungarian matching on the contingency table,
for scoring only. No labels used during fitting, no text at all -- this
is the label-free counterpart to embedding_knn_probe*.py's supervised
kNN probe, added to isolate whether the zero-shot text-decision layer
specifically (not supervision in general) is the bottleneck. See
ISBI_EXPERIMENT_SPEC.md Sec. 5.

Generalizes over binary (pathology) and multi-class (organ-ID, up to
24-way vertebral level) tasks alike -- unlike the AUC-based kNN probe
scripts, this reports balanced accuracy (primary), plus ARI and NMI
(both assignment-free, i.e. valid even without the Hungarian step, useful
as a robustness check on the Hungarian-matched balanced accuracy).

Reads the same cached .npz files the zero-shot runners already produce
(`pc_features`, `gt_labels`, plus zero-shot's own `preds`/`logits` for the
side-by-side ZS-vs-Clust comparison) -- no new feature extraction needed
to run this against any task that already has cached results.

Usage:
  python clustering_probe.py                          # run every CONFIGS entry
  python clustering_probe.py --npz results_kidney.npz  # single file, ad-hoc
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi')
N_INIT = 10
SEED = 0

# (label, npz path relative to ROOT, task name) -- extend as new
# model x task result files land. Kept deliberately explicit (not
# glob-derived) so a probe run is always over a known, reviewed set of
# files, matching the convention in embedding_knn_probe*.py.
CONFIGS = [
    ('ULIP-2', 'results_kidney.npz', 'Kidney tumor'),
    ('ULIP-2', 'results_brain.npz', 'Brain tumor'),
    ('ULIP-2', 'results_organid_ulip2.npz', 'Organ-ID (9-way)'),
    ('OpenShape', 'results_openshape_kidney.npz', 'Kidney tumor'),
    ('OpenShape', 'results_openshape_brain.npz', 'Brain tumor'),
    ('OpenShape', 'results_organid_openshape.npz', 'Organ-ID (9-way)'),
    ('PointCLIP V2', 'results_pointclipv2_kidney.npz', 'Kidney tumor'),
    ('PointCLIP V2', 'results_pointclipv2_brain.npz', 'Brain tumor'),
    ('PointCLIP V2', 'results_organid_pointclipv2.npz', 'Organ-ID (9-way)'),
    ('Uni3D', 'results_uni3d_kidney.npz', 'Kidney tumor'),
    ('Uni3D', 'results_uni3d_brain.npz', 'Brain tumor'),
    ('Uni3D', 'results_organid_uni3d.npz', 'Organ-ID (9-way)'),
    ('Point-Bind', 'results_pointbind_kidney.npz', 'Kidney tumor'),
    ('Point-Bind', 'results_pointbind_brain.npz', 'Brain tumor'),
    ('Point-Bind', 'results_organid_pointbind.npz', 'Organ-ID (9-way)'),
    ('JM3D', 'results_jm3d_kidney.npz', 'Kidney tumor'),
    ('JM3D', 'results_jm3d_brain.npz', 'Brain tumor'),
    ('JM3D', 'results_organid_jm3d.npz', 'Organ-ID (9-way)'),
]


def clustering_metrics(feats, gt, n_init=N_INIT, seed=SEED):
    """Fit k-means (k = #ground-truth classes) on standardized features,
    with no labels involved in fitting. Map clusters to classes via
    Hungarian matching on the contingency table (maximizes raw
    agreement) for scoring only, then report balanced accuracy plus the
    two assignment-free clustering metrics (ARI, NMI) as a check that
    the Hungarian-matched number isn't an artifact of the matching step
    itself."""
    n_classes = len(np.unique(gt))
    feats_s = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=n_classes, n_init=n_init, random_state=seed)
    cluster_ids = km.fit_predict(feats_s)

    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    for c, y in zip(cluster_ids, gt):
        contingency[c, y] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = dict(zip(row_ind, col_ind))
    mapped_preds = np.array([mapping[c] for c in cluster_ids])

    return dict(
        clust_bal_acc=balanced_accuracy_score(gt, mapped_preds),
        clust_acc=(mapped_preds == gt).mean(),
        clust_ari=adjusted_rand_score(gt, cluster_ids),
        clust_nmi=normalized_mutual_info_score(gt, cluster_ids),
        mapped_preds=mapped_preds,
        cluster_ids=cluster_ids,
    )


def zero_shot_metrics(gt, zs_preds):
    """Zero-shot needs no held-out split (no fitting happens at all), so
    it's scored on the full set directly -- no fold-averaging, no
    leakage concern, matches how it's scored in clustering_metrics'
    Clust-vs-ZS comparison."""
    return dict(
        zs_bal_acc=balanced_accuracy_score(gt, zs_preds),
        zs_acc=(zs_preds == gt).mean(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=None,
                     help='Run a single .npz file ad-hoc instead of CONFIGS.')
    ap.add_argument('--root', default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    if args.npz:
        configs = [('ad-hoc', args.npz, Path(args.npz).stem)]
    else:
        configs = CONFIGS

    print(f'{"Model":<14}{"Task":<18}{"ZS BalAcc":<11}{"Clust BalAcc":<13}'
          f'{"Clust ARI":<11}{"Clust NMI":<11}{"Higher":<8}')
    rows = []
    for model_name, fname, task in configs:
        path = root / fname
        if not path.exists():
            print(f'  SKIP (missing): {path}')
            continue
        data = np.load(path, allow_pickle=True)
        feats = data['pc_features']
        gt = data['gt_labels']

        cm = clustering_metrics(feats, gt)
        zm = zero_shot_metrics(gt, data['preds']) if 'preds' in data else {}

        gap = cm['clust_bal_acc'] - zm.get('zs_bal_acc', float('nan'))
        higher = 'Clust' if gap > 0 else ('ZS' if gap < 0 else 'tie')
        print(f'{model_name:<14}{task:<18}'
              f'{zm.get("zs_bal_acc", float("nan")):<11.4f}'
              f'{cm["clust_bal_acc"]:<13.4f}{cm["clust_ari"]:<11.4f}'
              f'{cm["clust_nmi"]:<11.4f}{higher:<8}')

        rows.append(dict(model=model_name, task=task, file=fname,
                          zs_bal_acc=zm.get('zs_bal_acc'),
                          clust_bal_acc=cm['clust_bal_acc'],
                          clust_ari=cm['clust_ari'], clust_nmi=cm['clust_nmi'],
                          gap=gap))

    out = root / 'clustering_probe_results.npy'
    np.save(out, rows, allow_pickle=True)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
