"""
Phase 0 of the proposed-fix work (few-shot cache-adapter, CoOp-style soft
prompts, zero-shot+clustering consensus): populate feature_cache/ with clean-
condition encoded features for all 6 models across all 21 tasks.

run_robustness_condition.py's run_one() computes point-cloud features in
memory and discards them after scoring -- only the scalar metric gets saved,
so none of the three new methods had anything to build on without
re-encoding from scratch. Feature caching (_cache_clean_features) was wired
into that file this session, but a (model, task) cell only gets cached when
something actually calls run_one(..., axis='clean') for it -- true so far
only for PointCLIP V2 on the 7 robustness-protocol tasks (today's prompt
ablation). This script explicitly drives every (model, task) cell once,
building each model ONLY ONCE (not once per task) to amortize the
multi-GB checkpoint load, then reuses run_robustness_condition.py's own
_cache_clean_features so the output format is identical to what that file
would have produced.

Usage:
  python populate_feature_cache.py --model ulip2
  python populate_feature_cache.py --model all
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import task_definitions as td
from run_robustness_condition import MODEL_LIB, FEATURE_CACHE_ROOT, _cache_clean_features


def build_point_model(model_key, lib, device):
    """Mirrors run_one()'s build dispatch: every lib except openshape
    returns a single object usable directly by encode_pointclouds; openshape
    returns (model, clip_model, tokenizer) and only `model` is needed here
    (text encoding isn't part of this script)."""
    built = lib.build_model(device=device)
    if model_key == 'openshape':
        model, _clip_model, _tokenizer = built
        return model
    return built


def run_model(model_key, device='cuda', batch_size=8, task_filter=None):
    lib = importlib.import_module(MODEL_LIB[model_key])
    print(f'[{model_key}] building model...')
    t0 = time.time()
    model = build_point_model(model_key, lib, device)
    print(f'[{model_key}] built in {time.time()-t0:.1f}s')

    tasks = ({task_filter: td.ALL_TASKS[task_filter]} if task_filter
              else td.ALL_TASKS)
    for task_name, task in tasks.items():
        out_path = FEATURE_CACHE_ROOT / f'{task_name}__{model_key}__clean.npz'
        if out_path.exists():
            print(f'[{model_key}/{task_name}] already cached, skipping')
            continue

        samples = td.collect_samples_multiroot(task['folder_label_root'])
        if not samples:
            print(f'[{model_key}/{task_name}] SKIP: no samples found')
            continue

        t0 = time.time()
        feats, gt, paths = lib.encode_pointclouds(
            model, samples, batch_size=batch_size, num_workers=8, device=device)
        feats = feats.cpu().numpy() if hasattr(feats, 'cpu') else np.asarray(feats)
        gt = gt.cpu().numpy() if hasattr(gt, 'cpu') else np.asarray(gt)
        dt = time.time() - t0
        print(f'[{model_key}/{task_name}] {len(samples)} samples encoded in {dt:.1f}s '
              f'({len(samples)/dt:.1f} samples/s)')

        _cache_clean_features(task_name, model_key, feats, gt, paths)
        print(f'[{model_key}/{task_name}] cached -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                     choices=list(MODEL_LIB.keys()) + ['all'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--task', default=None, choices=list(td.ALL_TASKS.keys()),
                     help='Limit to one task (smoke-testing only).')
    args = ap.parse_args()

    device = 'cuda'
    models = list(MODEL_LIB.keys()) if args.model == 'all' else [args.model]

    for m in models:
        print('=' * 70)
        print(f'>>> model={m}')
        print('=' * 70)
        run_model(m, device=device, batch_size=args.batch_size, task_filter=args.task)


if __name__ == '__main__':
    main()
