"""
Proposed fix: cache-adapter with disagreement-informed few-shot selection,
instead of random K-shot sampling.

Motivation: the paper's whole argument rests on a diagnostic gap between
two signals computed on the same frozen embedding -- zero-shot (uses text,
often wrong) and k-means clustering (no text, no labels, usually closer to
the truth). Random K-shot sampling for the cache-adapter ignores this gap
entirely. This asks: if you can only afford to label K examples per class,
should you pick them randomly, or specifically from where the broken
component (zero-shot) disagrees with the working one (clustering)?

Non-circularity matters here: Hungarian-matching clusters to class NAMES
requires ground truth, so using that mapping as the selection signal would
quietly need labels beyond the K being selected. Instead, the disagreement
signal here is fully label-free: for each cluster (unsupervised, no labels
needed to just assign samples to clusters), compute the MAJORITY zero-shot
prediction among its members (using zero-shot's own predictions, not
ground truth), then flag a sample as "disagreement" if its own zero-shot
prediction differs from its cluster's majority zero-shot vote -- i.e. the
text layer is calling this sample something different from what it calls
most of the samples the representation groups it with. Ground truth is
only spent on the K examples actually selected, exactly like the random
baseline -- same labeling budget, different (informed vs random) choice of
which K to spend it on.

Usage:
  python informed_cache_adapter.py --model ulip2
  python informed_cache_adapter.py --model all
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
from cache_adapter import K_SHOTS, SEEDS, ALPHA, BETA, build_text_model

OUT_DIR = Path('/msc/home/q45emuni/miccai_2026/shapemi/scripts/informed_cache_adapter_results')
N_INIT = 10
CLUSTER_SEED = 0


def disagreement_kshot_indices(gt, zs_preds, cluster_ids, k, n_classes, seed):
    """Label-free disagreement signal (zero-shot pred vs cluster's own
    majority zero-shot vote), then spend the K-per-class label budget on
    disagreement examples first, falling back to random fill if a class
    doesn't have enough disagreement examples."""
    n_clusters = cluster_ids.max() + 1
    cluster_mode_pred = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        members = zs_preds[cluster_ids == c]
        if len(members) > 0:
            vals, counts = np.unique(members, return_counts=True)
            cluster_mode_pred[c] = vals[np.argmax(counts)]
    disagreement = zs_preds != cluster_mode_pred[cluster_ids]

    rng = np.random.RandomState(seed)
    support_idx = []
    for c in range(n_classes):
        class_idx = np.where(gt == c)[0]
        if len(class_idx) < k:
            return None
        disagree_in_class = class_idx[disagreement[class_idx]]
        agree_in_class = class_idx[~disagreement[class_idx]]
        if len(disagree_in_class) >= k:
            chosen = rng.choice(disagree_in_class, size=k, replace=False)
        else:
            chosen = disagree_in_class
            n_more = k - len(chosen)
            fill = rng.choice(agree_in_class, size=n_more, replace=False)
            chosen = np.concatenate([chosen, fill])
        support_idx.extend(chosen.tolist())
    return np.array(support_idx)


def run_task(encode_text_fn, task_name, task, device='cuda'):
    cache_path = FEATURE_CACHE_ROOT / f'{task_name}__{encode_text_fn.model_key}__clean.npz'
    d = np.load(cache_path, allow_pickle=True)
    feats = torch.from_numpy(d['feats']).float().to(device)
    gt = d['gt']
    n_samples = len(gt)
    n_classes = len(task['class_names'])

    text_features = encode_text_fn(task['class_names']).to(device).float()
    zs_logits = (feats @ text_features.t()).cpu().numpy()
    zs_preds = zs_logits.argmax(axis=1)
    zs_bal_acc = balanced_accuracy_score(gt, zs_preds)

    feats_np = feats.cpu().numpy()
    feats_s = StandardScaler().fit_transform(feats_np)
    km = KMeans(n_clusters=n_classes, n_init=N_INIT, random_state=CLUSTER_SEED)
    cluster_ids = km.fit_predict(feats_s)

    rows = []
    for k in K_SHOTS:
        for seed in SEEDS:
            support_idx = disagreement_kshot_indices(gt, zs_preds, cluster_ids, k, n_classes, seed)
            if support_idx is None:
                continue
            n_disagree_selected = int(np.sum(zs_preds[support_idx] !=
                                              np.array([np.bincount(zs_preds[cluster_ids == cluster_ids[i]]).argmax()
                                                         for i in support_idx])))

            query_mask = np.ones(n_samples, dtype=bool)
            query_mask[support_idx] = False
            query_idx = np.where(query_mask)[0]

            keys = feats[support_idx]
            values = np.zeros((len(support_idx), n_classes), dtype=np.float32)
            values[np.arange(len(support_idx)), gt[support_idx]] = 1.0
            values_t = torch.from_numpy(values).to(device)

            query_feats = feats[query_idx]
            cos_sim = query_feats @ keys.t()
            affinity = torch.exp(-BETA * (1 - cos_sim))
            cache_logits = (affinity @ values_t).cpu().numpy()

            final_logits = zs_logits[query_idx] + ALPHA * cache_logits
            preds = final_logits.argmax(axis=1)
            bal_acc = balanced_accuracy_score(gt[query_idx], preds)
            zs_query_bal_acc = balanced_accuracy_score(gt[query_idx], zs_logits[query_idx].argmax(axis=1))

            rows.append(dict(
                task=task_name, k=k, seed=seed, n_samples=n_samples, n_classes=n_classes,
                n_query=len(query_idx), zs_bal_acc_full=zs_bal_acc, zs_bal_acc_query=zs_query_bal_acc,
                informed_cache_bal_acc=bal_acc, delta=bal_acc - zs_query_bal_acc,
                n_disagree_selected=n_disagree_selected, n_selected=len(support_idx),
            ))
    return rows


def run_model(model_key, device='cuda', task_filter=None):
    lib = importlib.import_module(MODEL_LIB[model_key])
    print(f'[{model_key}] building model for text encoding...')
    encode_text_fn = build_text_model(model_key, lib, device)
    encode_text_fn.model_key = model_key

    all_rows = []
    tasks = ({task_filter: td.ALL_TASKS[task_filter]} if task_filter
             else td.ALL_TASKS)
    for task_name, task in tasks.items():
        rows = run_task(encode_text_fn, task_name, task, device=device)
        for r in rows:
            r['model'] = model_key
        all_rows.extend(rows)
        if rows:
            by_k = {}
            for r in rows:
                by_k.setdefault(r['k'], []).append(r['informed_cache_bal_acc'])
            summary = ', '.join(f'K={k}:{np.mean(v):.3f}' for k, v in sorted(by_k.items()))
            frac_disagree = rows[-1]['n_disagree_selected'] / rows[-1]['n_selected']
            print(f'[{model_key}/{task_name}] {summary}  (K=16 disagree-frac={frac_disagree:.2f})')
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_LIB.keys()) + ['all'])
    ap.add_argument('--task', default=None, choices=list(td.ALL_TASKS.keys()))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = list(MODEL_LIB.keys()) if args.model == 'all' else [args.model]

    fieldnames = ['model', 'task', 'k', 'seed', 'n_samples', 'n_classes', 'n_query',
                  'zs_bal_acc_full', 'zs_bal_acc_query', 'informed_cache_bal_acc', 'delta',
                  'n_disagree_selected', 'n_selected']
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for m in models:
        print('=' * 70)
        print(f'>>> model={m}')
        print('=' * 70)
        rows = run_model(m, device=device, task_filter=args.task)
        out_path = OUT_DIR / f'{m}.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()
