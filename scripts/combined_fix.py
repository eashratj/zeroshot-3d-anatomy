"""
Proposed fix: fuse the few-shot cache-adapter (Tip-Adapter-style, training-
free) with soft prompt tuning (CoOp-style, learned context vectors).

Motivation: run independently, these two few-shot strategies have
complementary failure modes. Cache-adapter is a non-parametric lookup --
robust everywhere, including the hardest task (T5 vertebral level, 24-way),
but caps out lower than soft prompt tuning on several easier tasks.
Soft prompt tuning matches or beats cache-adapter on most tasks but
essentially fails to learn at all on T5 vertebral level (mean bal_acc
~0.21 at K=16, vs cache-adapter's ~0.55) -- plausibly because fitting 16
learned context vectors via gradient descent on as few as 24 total K=1
examples across 24 classes is a genuine optimization-failure regime a
non-parametric method doesn't share.

Fusion: average the two methods' softmax probability distributions (equal,
untrained 50/50 weight -- not tuned against test accuracy, same discipline
used for the clustering-consensus method's lambda=0.5 default). Both
methods already cost <0.15ms/query at inference (measured separately), so
fusing them adds no meaningful inference-time overhead.

Zero-shot and soft-prompt logits come from two genuinely separate text-
encoding pathways (generic hand-written TEMPLATES vs. the learned CoOp
context), so this builds both: a text-only zero-shot encoder (same pattern
as cluster_consensus.py's build_text_model) and a CoOp prompt-learner
factory (soft_prompt_tuning.py's build_prompt_learner) -- each loads its
own copy of the backbone, roughly 2x the one-time load cost of either
method alone, acceptable given the deadline.

Reuses the exact same stratified_kshot_indices(gt, k, n_classes, seed) draw
as both standalone methods (deterministic given the same seed), so results
are directly comparable cell-for-cell against both prior runs.

Usage:
  python combined_fix.py --model ulip2
  python combined_fix.py --model all
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import task_definitions as td
from run_robustness_condition import MODEL_LIB, FEATURE_CACHE_ROOT
from cache_adapter import K_SHOTS, SEEDS, ALPHA, BETA, stratified_kshot_indices
from cluster_consensus import build_text_model
from soft_prompt_tuning import build_prompt_learner, train_and_get_logits

OUT_DIR = Path('/msc/home/q45emuni/miccai_2026/shapemi/scripts/combined_fix_results')


def cache_adapter_logits(feats, gt, support_idx, query_idx, zs_logits, n_classes, device):
    keys = feats[support_idx]
    values = np.zeros((len(support_idx), n_classes), dtype=np.float32)
    values[np.arange(len(support_idx)), gt[support_idx]] = 1.0
    values_t = torch.from_numpy(values).to(device)

    query_feats = feats[query_idx]
    cos_sim = query_feats @ keys.t()
    affinity = torch.exp(-BETA * (1 - cos_sim))
    cache_logits = (affinity @ values_t).cpu().numpy()
    return zs_logits[query_idx] + ALPHA * cache_logits


def run_model(model_key, device='cuda', task_filter=None):
    lib = importlib.import_module(MODEL_LIB[model_key])
    print(f'[{model_key}] building text-only zero-shot encoder...')
    encode_text_fn = build_text_model(model_key, lib, device)

    all_rows = []
    tasks = ({task_filter: td.ALL_TASKS[task_filter]} if task_filter
             else td.ALL_TASKS)
    for task_name, task in tasks.items():
        cache_path = FEATURE_CACHE_ROOT / f'{task_name}__{model_key}__clean.npz'
        d = np.load(cache_path, allow_pickle=True)
        feats_t = torch.from_numpy(d['feats']).float().to(device)
        gt = d['gt']
        n_samples = len(gt)
        n_classes = len(task['class_names'])

        text_features = encode_text_fn(task['class_names']).to(device).float()
        zs_logits_np = (feats_t @ text_features.t()).cpu().numpy()

        prompt_factory = build_prompt_learner(model_key, lib, task['class_names'], device)
        prompt_factory.model_key = model_key

        rows = []
        for k in K_SHOTS:
            for seed in SEEDS:
                support_idx = stratified_kshot_indices(gt, k, n_classes, seed)
                if support_idx is None:
                    continue
                query_mask = np.ones(n_samples, dtype=bool)
                query_mask[support_idx] = False
                query_idx = np.where(query_mask)[0]

                sp_logits = train_and_get_logits(
                    prompt_factory, feats_t, gt, task['class_names'], support_idx, query_idx, device)
                sp_preds = sp_logits.argmax(axis=1)
                sp_bal_acc = balanced_accuracy_score(gt[query_idx], sp_preds)

                ca_logits = cache_adapter_logits(
                    feats_t, gt, support_idx, query_idx, zs_logits_np, n_classes, device)
                ca_preds = ca_logits.argmax(axis=1)
                ca_bal_acc = balanced_accuracy_score(gt[query_idx], ca_preds)

                ca_probs = F.softmax(torch.from_numpy(ca_logits), dim=1).numpy()
                sp_probs = F.softmax(torch.from_numpy(sp_logits), dim=1).numpy()
                fused_probs = 0.5 * ca_probs + 0.5 * sp_probs
                fused_preds = fused_probs.argmax(axis=1)
                fused_bal_acc = balanced_accuracy_score(gt[query_idx], fused_preds)

                zs_bal_acc = balanced_accuracy_score(
                    gt[query_idx], zs_logits_np[query_idx].argmax(axis=1))

                rows.append(dict(
                    task=task_name, k=k, seed=seed, n_samples=n_samples, n_classes=n_classes,
                    n_query=len(query_idx), zs_bal_acc=zs_bal_acc,
                    cache_adapter_bal_acc=ca_bal_acc, soft_prompt_bal_acc=sp_bal_acc,
                    fused_bal_acc=fused_bal_acc,
                ))
        del prompt_factory
        torch.cuda.empty_cache()

        for r in rows:
            r['model'] = model_key
        all_rows.extend(rows)
        if rows:
            by_k = {}
            for r in rows:
                by_k.setdefault(r['k'], []).append(
                    (r['cache_adapter_bal_acc'], r['soft_prompt_bal_acc'], r['fused_bal_acc']))
            summary = ', '.join(
                f'K={k}: CA={np.mean([v[0] for v in vs]):.3f} '
                f'SP={np.mean([v[1] for v in vs]):.3f} '
                f'Fused={np.mean([v[2] for v in vs]):.3f}'
                for k, vs in sorted(by_k.items()))
            print(f'[{model_key}/{task_name}] {summary}')
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_LIB.keys()) + ['all'])
    ap.add_argument('--task', default=None, choices=list(td.ALL_TASKS.keys()))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = list(MODEL_LIB.keys()) if args.model == 'all' else [args.model]

    fieldnames = ['model', 'task', 'k', 'seed', 'n_samples', 'n_classes', 'n_query',
                  'zs_bal_acc', 'cache_adapter_bal_acc', 'soft_prompt_bal_acc', 'fused_bal_acc']
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
