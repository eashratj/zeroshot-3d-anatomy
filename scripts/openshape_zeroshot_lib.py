"""
Shared utilities for OpenShape zero-shot classification on MedShapeNet point
clouds. Mirrors ulip_zeroshot_lib.py's interface so run_experiment_openshape.py
can reuse the same experiment configs and metric computation.

Uses the PointBERT (ppat) backbone + OpenCLIP ViT-bigG-14 text tower, since the
SparseConv variant needs MinkowskiEngine (avoided, see models/__init__.py patch).
Reuses the exact same cached point clouds as ULIP-2: verified both repos use
the identical normalize_pc convention (center + divide by max L2 norm) and the
same colorless-fill convention (rgb = 0.4 constant).
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

OPENSHAPE_ROOT = '/msc/home/q45emuni/miccai_2026/OpenShape/src'
if OPENSHAPE_ROOT not in sys.path:
    sys.path.insert(0, OPENSHAPE_ROOT)

CHECKPOINT_REPO = 'OpenShape/openshape-pointbert-vitg14-rgb'
POINTCLOUD_ROOT = Path('/msc/home/q45emuni/foundation_models/shapemi/pointclouds')

TEMPLATES = [
    "a 3D model of a {}",
    "an anatomical shape of a {}",
    "a medical segmentation of a {}",
    "a point cloud of a {}",
    "a 3D scan of a {}",
]

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data


def _make_config():
    return SimpleNamespace(
        clip_embed_dim=1280,
        model=SimpleNamespace(
            name='PointBERT',
            scaling=4,
            in_channel=6,
            out_channel=1280,
        ),
    )


def build_model(device='cuda'):
    import models
    from collections import OrderedDict
    from huggingface_hub import hf_hub_download

    config = _make_config()
    model = models.make(config).to(device)

    ckpt_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename='model.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict)
    model.eval()

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

    return model, clip_model, tokenizer


@torch.no_grad()
def encode_class_text(clip_model, tokenizer, class_names, device='cuda'):
    text_features = []
    for name in class_names:
        texts = [t.format(name) for t in TEMPLATES]
        tokens = tokenizer(texts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        emb = emb.mean(dim=0)
        emb = F.normalize(emb, dim=0)
        text_features.append(emb)
    return torch.stack(text_features, dim=0)  # (num_classes, 1280)


class PointCloudDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path).astype(np.float32)  # (10000, 6) xyzrgb
        return torch.from_numpy(pc), label, str(npy_path)


def collect_samples(folder_to_label: dict):
    samples = []
    for folder, label in folder_to_label.items():
        folder_dir = POINTCLOUD_ROOT / folder
        files = sorted(folder_dir.glob('*.npy'))
        for f in files:
            samples.append((f, label))
    return samples


@torch.no_grad()
def encode_pointclouds(model, samples, batch_size=32, num_workers=4, device='cuda', yz_swap=False):
    dataset = PointCloudDataset(samples)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    all_feats, all_labels, all_paths = [], [], []
    for pc, label, paths in loader:
        pc = pc.to(device, non_blocking=True)
        if yz_swap:
            # OpenShape's training data uses a Y-up gravity-axis convention
            # (only z-axis rotation is randomized during training, not axis
            # choice -- see dataset.random_z_rotate in their train.yaml).
            # MedShapeNet meshes have no such canonical orientation; this
            # swaps Y/Z to match their convention, as their own demo
            # (src/example.py::load_ply) does for arbitrary z-up input.
            # Empirically ablated per-organ since results are mixed.
            pc = pc.clone()
            pc[:, :, [1, 2]] = pc[:, :, [2, 1]]
        xyz = pc[:, :, :3].contiguous()
        feat = pc.contiguous()  # full xyzrgb (6ch), matching model.in_channel=6
        shape_feat = model(xyz, feat)
        shape_feat = F.normalize(shape_feat, dim=-1)
        all_feats.append(shape_feat.cpu())
        all_labels.append(label)
        all_paths.extend(paths)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0), all_paths
