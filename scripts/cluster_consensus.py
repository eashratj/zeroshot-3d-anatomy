"""
Clustering consensus: zero-shot + clustering consensus re-scoring. Fully
unsupervised (0-shot) -- no labels anywhere, no new training.

Motivation: the paper already shows clustering recovers near-kNN accuracy on
the same frozen embeddings with zero labels and zero text, while zero-shot
(same embeddings, uses text) often collapses near chance. This blends the
two signals: for each sample, average its own zero-shot logits with the
*cluster-average* zero-shot logits of its k-means cluster (Hungarian matching
is NOT used here -- that would require ground truth; this only uses each
cluster's own aggregate zero-shot score, nothing from clustering_probe.py's
label-dependent scoring path).

final_logits = (1-lambda) * sample_zeroshot_logits + lambda * cluster_mean_zeroshot_logits

lambda must not be tuned against test accuracy (that would leak supervision
through the back door for a method whose whole point is 0-shot). lambda=0.5
is reported as the untuned headline number; the full sweep is a diagnostic
sensitivity check only, not a selection procedure.

Usage:
  python cluster_consensus.py --model ulip2
  python cluster_consensus.py --model all
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import task_definitions as td
from run_robustness_condition import MODEL_LIB, FEATURE_CACHE_ROOT

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HEADLINE_LAMBDA = 0.5
OUT_DIR = Path('/msc/home/q45emuni/miccai_2026/shapemi/scripts/cluster_consensus_results')
N_INIT = 10
SEED = 0


def build_text_model(model_key, lib, device):
    """Mirrors run_one()'s build dispatch for text encoding: openshape
    needs (clip_model, tokenizer) separately from the point model, every
    other lib's encode_class_text takes the single built object."""
    built = lib.build_model(device=device)
    if model_key == 'openshape':
        _model, clip_model, tokenizer = built
        return lambda class_names: lib.encode_class_text(
            clip_model, tokenizer, class_names, device=device)
    return lambda class_names: lib.encode_class_text(built, class_names, device=device)


def run_task(encode_text_fn, task_name, task, device='cuda'):
    cache_path = FEATURE_CACHE_ROOT / f'{task_name}__{encode_text_fn.model_key}__clean.npz'
    d = np.load(cache_path, allow_pickle=True)
    feats = torch.from_numpy(d['feats']).float()
    gt = d['gt']

    text_features = encode_text_fn(task['class_names'])
    text_features = text_features.to(device).float()

    zs_logits = (feats.to(device) @ text_features.t()).cpu().numpy()
    zs_preds = zs_logits.argmax(axis=1)
    zs_bal_acc = balanced_accuracy_score(gt, zs_preds)

    n_classes = len(task['class_names'])
    feats_np = feats.numpy()
    feats_s = StandardScaler().fit_transform(feats_np)
    km = KMeans(n_clusters=n_classes, n_init=N_INIT, random_state=SEED)
    cluster_ids = km.fit_predict(feats_s)

    cluster_mean_logits = np.zeros((n_classes, n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask = cluster_ids == c
        if mask.sum() > 0:
            cluster_mean_logits[c] = zs_logits[mask].mean(axis=0)
    per_sample_cluster_logits = cluster_mean_logits[cluster_ids]

    row = dict(task=task_name, n_samples=len(gt), n_classes=n_classes,
               zs_bal_acc=zs_bal_acc)
    for lam in LAMBDAS:
        blended = (1 - lam) * zs_logits + lam * per_sample_cluster_logits
        preds = blended.argmax(axis=1)
        row[f'method3_bal_acc_lambda{lam}'] = balanced_accuracy_score(gt, preds)
    return row


def run_model(model_key, device='cuda'):
    lib = importlib.import_module(MODEL_LIB[model_key])
    print(f'[{model_key}] building model for text encoding...')
    encode_text_fn = build_text_model(model_key, lib, device)
    encode_text_fn.model_key = model_key

    rows = []
    for task_name, task in td.ALL_TASKS.items():
        row = run_task(encode_text_fn, task_name, task, device=device)
        row['model'] = model_key
        rows.append(row)
        headline = row[f'method3_bal_acc_lambda{HEADLINE_LAMBDA}']
        delta = headline - row['zs_bal_acc']
        print(f'[{model_key}/{task_name}] ZS={row["zs_bal_acc"]:.4f} '
              f'Method3(λ=0.5)={headline:.4f} delta={delta:+.4f}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                     choices=list(MODEL_LIB.keys()) + ['all'])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = list(MODEL_LIB.keys()) if args.model == 'all' else [args.model]

    fieldnames = ['model', 'task', 'n_samples', 'n_classes', 'zs_bal_acc'] + \
                 [f'method3_bal_acc_lambda{lam}' for lam in LAMBDAS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for m in models:
        print('=' * 70)
        print(f'>>> model={m}')
        print('=' * 70)
        rows = run_model(m, device=device)
        out_path = OUT_DIR / f'{m}.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()
