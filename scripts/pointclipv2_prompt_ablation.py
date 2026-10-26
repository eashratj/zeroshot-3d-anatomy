"""
PointCLIP V2 prompt-mismatch ablation, on the current full-scale 277-cohort
data (not the SHAPEMI-pilot diagnostic this is modeled on -- see
native_prompt_diagnostic.py for the earlier, smaller-scope version).

Motivation (Limitations item 5 / reviewer discussion): PointCLIP V2 is the
only model in the study whose "shape encoder" is actually a 2D image
encoder (OpenAI CLIP ViT-B/16) run on rendered depth-map views, not a
point-cloud-native encoder. The project's shared TEMPLATES
(pointclipv2_zeroshot_lib.py) were written in point-cloud/3D-scan language
("a point cloud of a {}", "a 3D scan of a {}") for consistency across all
six models -- but that phrasing describes a representation PointCLIP V2's
*text* encoder never actually sees a matching image for: its image
encoder only ever sees a 2D depth-map render. This script asks how much
of PointCLIP V2's zero-shot failure is attributable to that mismatch,
holding the point-cloud encoding (and therefore the depth-map renders)
completely fixed and varying ONLY the text-prompt templates used to
encode the class names.

Two prompt conditions, scored from the SAME cached image features (so the
comparison isolates the text side exactly):
  - generic: the project's own 5-template set, used throughout the paper.
  - depth-map (native-style): the 3-template approximation of PointCLIP
    V2's own GPT-3-generated per-class prompt style (their released
    prompts are searched specifically for ModelNet40 classes we don't
    have entries for, so this is an approximation of their phrasing
    convention, not their exact searched prompts -- same approximation
    and same caveat as native_prompt_diagnostic.py's SHAPEMI-pilot
    version, reused here unchanged for continuity).

Run on the same 7 representative task groups as Table 1 / the robustness
sweep (T2 laterality via its kidney structure, standing in for all 15).

Usage:
  python pointclipv2_prompt_ablation.py
  python pointclipv2_prompt_ablation.py --task T1_organ_id   # single task
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

import task_definitions as td
from pointclipv2_zeroshot_lib import build_model, collect_samples, encode_pointclouds, TEMPLATES

REPRESENTATIVE_TASKS = [
    'T1_organ_id', 'T2_kidney', 'T3_lung_lobes', 'T4_cardiac',
    'T5_vertebrae', 'T6_kidney_pathology', 'T6_brain_pathology',
]

DEPTH_MAP_TEMPLATES = [
    "The depth map is a 3D representation of a {}.",
    "This is a depth map of a 3D model of a {}.",
    "This is a depth map of a {} 3D model.",
]

OUT_CSV = Path('/msc/home/q45emuni/miccai_2026/shapemi/scripts/pointclipv2_prompt_ablation_results.csv')

# Shared with run_robustness_condition.py's clean-condition cache -- same
# samples, same encoder, same features, so a prompt-only ablation like this
# one should never force a second multi-view re-render from scratch.
FEATURE_CACHE_ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi/feature_cache')


def _load_or_encode_features(model, task_name, samples, batch_size, device):
    cache_path = FEATURE_CACHE_ROOT / f'{task_name}__pointclipv2__clean.npz'
    if cache_path.exists():
        print(f'[{task_name}] loading cached features from {cache_path}')
        d = np.load(cache_path, allow_pickle=True)
        return torch.from_numpy(d['feats']), torch.from_numpy(d['gt'])
    feats, gt, paths = encode_pointclouds(model, samples, batch_size=batch_size, device=device)
    FEATURE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # np.savez() appends '.npz' to any path not already ending with it, so
    # a '*.npz.tmp' name actually lands as '*.npz.tmp.npz' -- keep the tmp
    # name ending in '.npz' itself.
    tmp_path = cache_path.with_name(cache_path.stem + '.tmp.npz')
    np.savez(tmp_path, feats=feats.numpy(), gt=gt.numpy(), paths=np.array(paths, dtype=object))
    tmp_path.rename(cache_path)
    return feats, gt


@torch.no_grad()
def encode_text_with_templates(model, class_names, templates, device='cuda'):
    clip_model = model['clip_model']
    tokenizer = model['tokenizer']
    text_features = []
    for name in class_names:
        texts = [t.format(name) for t in templates]
        tokens = tokenizer(texts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        emb = emb / emb.norm(dim=0)
        text_features.append(emb)
    return torch.stack(text_features, dim=0).float()


def run_task(model, task_name, batch_size, device='cuda'):
    task = td.ALL_TASKS[task_name]
    samples = td.collect_samples_multiroot(task['folder_label_root'])
    class_names = task['class_names']
    print(f'[{task_name}] {len(samples)} samples, {len(class_names)} classes')

    t0 = time.time()
    feats, gt = _load_or_encode_features(model, task_name, samples, batch_size, device)
    print(f'[{task_name}] features ready in {time.time()-t0:.1f}s')

    gt = gt.numpy()
    results = {}
    for cond_name, templates in [('generic', TEMPLATES), ('depth_map', DEPTH_MAP_TEMPLATES)]:
        text_features = encode_text_with_templates(model, class_names, templates, device=device)
        logits = (feats.to(device) @ text_features.to(device).t()).cpu().numpy()
        preds = logits.argmax(axis=1)
        bal_acc = balanced_accuracy_score(gt, preds)
        results[cond_name] = bal_acc
        print(f'[{task_name}] {cond_name:10s} bal_acc={bal_acc:.4f}')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='all', choices=REPRESENTATIVE_TASKS + ['all'])
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('Building PointCLIP V2 model...')
    model = build_model(device=device)

    tasks = REPRESENTATIVE_TASKS if args.task == 'all' else [args.task]

    all_results = {}
    for t in tasks:
        print('=' * 70)
        all_results[t] = run_task(model, t, batch_size=args.batch_size, device=device)

    print()
    print(f'{"Task":22s} {"generic":>10s} {"depth_map":>10s} {"delta":>10s}')
    rows = []
    for t, r in all_results.items():
        delta = r['depth_map'] - r['generic']
        print(f'{t:22s} {r["generic"]:10.4f} {r["depth_map"]:10.4f} {delta:+10.4f}')
        rows.append({'task': t, 'generic_bal_acc': r['generic'],
                      'depth_map_bal_acc': r['depth_map'], 'delta': delta})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'generic_bal_acc', 'depth_map_bal_acc', 'delta'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {OUT_CSV}')


if __name__ == '__main__':
    main()
