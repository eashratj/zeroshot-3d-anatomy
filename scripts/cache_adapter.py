"""
Few-shot cache-adapter, Tip-Adapter-style. Few-shot
supervision (K in {1,2,4,8,16} labeled examples per class) -- the first of
the three proposed-fix methods to use actual labels, unlike the 0-shot
clustering-consensus method (Phase 1 / "Method 3" in the original plan
numbering, weak result: 24/42 combos improved, median delta 0.0000 at every
blend weight tried).

Standard Tip-Adapter (Zhang et al. 2022) formulation, applied to frozen
shape embeddings instead of 2D CLIP image embeddings: build a small "cache"
from K labeled examples per class (keys = their L2-normalized embeddings,
values = one-hot labels), score every other ("query") sample's affinity to
each cached key, and blend the resulting cache logits with the model's
existing zero-shot logits:

  affinity = exp(-beta * (1 - cosine_sim(query, keys)))
  cache_logits = affinity @ one_hot_values
  final_logits = zeroshot_logits + alpha * cache_logits

alpha=1.0, beta=5.5 are the original Tip-Adapter paper's defaults, used
here untuned (not searched against test accuracy) to keep this an honestly
training-free, lightweight method. 3 random seeds per K average out which
specific K examples got drawn (matters most at K=1/K=2).

Usage:
  python cache_adapter.py --model ulip2
  python cache_adapter.py --model all
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import task_definitions as td
from run_robustness_condition import MODEL_LIB, FEATURE_CACHE_ROOT

K_SHOTS = [1, 2, 4, 8, 16]
SEEDS = [0, 1, 2]
ALPHA = 1.0
BETA = 5.5
OUT_DIR = Path('/msc/home/q45emuni/miccai_2026/shapemi/scripts/cache_adapter_results')


def build_text_model(model_key, lib, device):
    """Same dispatch as cluster_consensus.py's build_text_model."""
    built = lib.build_model(device=device)
    if model_key == 'openshape':
        _model, clip_model, tokenizer = built
        return lambda class_names: lib.encode_class_text(
            clip_model, tokenizer, class_names, device=device)
    return lambda class_names: lib.encode_class_text(built, class_names, device=device)


def stratified_kshot_indices(gt, k, n_classes, seed):
    rng = np.random.RandomState(seed)
    support_idx = []
    for c in range(n_classes):
        class_idx = np.where(gt == c)[0]
        if len(class_idx) < k:
            return None  # not enough samples of this class for this K
        chosen = rng.choice(class_idx, size=k, replace=False)
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
    zs_bal_acc = balanced_accuracy_score(gt, zs_logits.argmax(axis=1))

    rows = []
    for k in K_SHOTS:
        for seed in SEEDS:
            support_idx = stratified_kshot_indices(gt, k, n_classes, seed)
            if support_idx is None:
                continue  # task doesn't have k samples/class -- skip this (k, seed)

            query_mask = np.ones(n_samples, dtype=bool)
            query_mask[support_idx] = False
            query_idx = np.where(query_mask)[0]

            keys = feats[support_idx]  # already L2-normalized by each model's encode_pointclouds
            values = np.zeros((len(support_idx), n_classes), dtype=np.float32)
            values[np.arange(len(support_idx)), gt[support_idx]] = 1.0
            values_t = torch.from_numpy(values).to(device)

            query_feats = feats[query_idx]
            cos_sim = query_feats @ keys.t()  # (n_query, k*n_classes)
            affinity = torch.exp(-BETA * (1 - cos_sim))
            cache_logits = (affinity @ values_t).cpu().numpy()

            final_logits = zs_logits[query_idx] + ALPHA * cache_logits
            preds = final_logits.argmax(axis=1)
            bal_acc = balanced_accuracy_score(gt[query_idx], preds)

            # zero-shot-only accuracy on the SAME query subset, for a fair
            # apples-to-apples comparison (excludes the K support examples
            # zero-shot never "sees" either, but keeps denominators matched)
            zs_query_bal_acc = balanced_accuracy_score(
                gt[query_idx], zs_logits[query_idx].argmax(axis=1))

            rows.append(dict(
                task=task_name, k=k, seed=seed, n_samples=n_samples,
                n_classes=n_classes, n_query=len(query_idx),
                zs_bal_acc_full=zs_bal_acc, zs_bal_acc_query=zs_query_bal_acc,
                method1_bal_acc=bal_acc, delta=bal_acc - zs_query_bal_acc,
            ))
    return rows


def run_model(model_key, device='cuda'):
    lib = importlib.import_module(MODEL_LIB[model_key])
    print(f'[{model_key}] building model for text encoding...')
    encode_text_fn = build_text_model(model_key, lib, device)
    encode_text_fn.model_key = model_key

    all_rows = []
    for task_name, task in td.ALL_TASKS.items():
        rows = run_task(encode_text_fn, task_name, task, device=device)
        for r in rows:
            r['model'] = model_key
        all_rows.extend(rows)
        if rows:
            by_k = {}
            for r in rows:
                by_k.setdefault(r['k'], []).append(r['method1_bal_acc'])
            summary = ', '.join(f'K={k}:{np.mean(v):.3f}' for k, v in sorted(by_k.items()))
            print(f'[{model_key}/{task_name}] ZS={rows[0]["zs_bal_acc_full"]:.4f}  {summary}')
        else:
            print(f'[{model_key}/{task_name}] SKIPPED (not enough samples/class for any K)')
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                     choices=list(MODEL_LIB.keys()) + ['all'])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = list(MODEL_LIB.keys()) if args.model == 'all' else [args.model]

    fieldnames = ['model', 'task', 'k', 'seed', 'n_samples', 'n_classes', 'n_query',
                  'zs_bal_acc_full', 'zs_bal_acc_query', 'method1_bal_acc', 'delta']
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
