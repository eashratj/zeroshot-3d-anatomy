"""
Shared utilities for ULIP-2 zero-shot classification on MedShapeNet point clouds.
"""

import os
import sys
from pathlib import Path

ULIP_ROOT = '/msc/home/q45emuni/miccai_2026/ULIP'
if ULIP_ROOT not in sys.path:
    sys.path.insert(0, ULIP_ROOT)
# ULIP2_PointBERT_Colored() loads its backbone config from a hardcoded path
# relative to the ULIP repo root ('./models/pointbert/...'), so cwd must match.
os.chdir(ULIP_ROOT)

CHECKPOINT_PATH = (
    '/msc/home/q45emuni/miccai_2026/ULIP/checkpoints/ULIP-2/pretrained_models/'
    'ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt'
)
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
import torch.utils.data as data


def build_model(device='cuda'):
    from models.ULIP_models import ULIP2_PointBERT_Colored

    class DummyArgs:
        pass

    model = ULIP2_PointBERT_Colored(DummyArgs())
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def encode_class_text(model, class_names, device='cuda'):
    """Ensemble-averaged, L2-normalized text embedding per class name."""
    tokenizer = model.tokenizer
    text_features = []
    for name in class_names:
        texts = [t.format(name) for t in TEMPLATES]
        tokens = tokenizer(texts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_features.append(emb)
    return torch.stack(text_features, dim=0)  # (num_classes, 1280)


class PointCloudDataset(data.Dataset):
    def __init__(self, samples):
        """samples: list of (npy_path, label_idx)"""
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path).astype(np.float32)  # (10000, 6)
        return torch.from_numpy(pc), label, str(npy_path)


def collect_samples(folder_to_label: dict):
    """folder_to_label: {organ_folder_name: label_idx}. Returns list of (npy_path, label_idx)."""
    samples = []
    for folder, label in folder_to_label.items():
        folder_dir = POINTCLOUD_ROOT / folder
        files = sorted(folder_dir.glob('*.npy'))
        for f in files:
            samples.append((f, label))
    return samples


@torch.no_grad()
def encode_pointclouds(model, samples, batch_size=32, num_workers=8, device='cuda'):
    dataset = PointCloudDataset(samples)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    all_feats, all_labels, all_paths = [], [], []
    for pc, label, paths in loader:
        pc = pc.to(device, non_blocking=True)
        feat = model.encode_pc(pc)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_feats.append(feat.cpu())
        all_labels.append(label)
        all_paths.extend(paths)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0), all_paths
