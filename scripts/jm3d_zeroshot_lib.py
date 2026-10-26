"""
Shared utilities for JM3D zero-shot classification on MedShapeNet point
clouds. Mirrors ulip_zeroshot_lib.py's interface exactly.

JM3D ("Beyond First Impressions: Integrating Joint Multi-modal Cues for
Comprehensive 3D Representation", ACM MM 2023, arXiv:2308.02982) is built
directly on the ULIP codebase (same models/ULIP_models.py, data/dataset_3d.py,
utils/tokenizer.py scaffolding as ULIP-2's own repo) -- we use its PointBERT
backbone variant (ULIP_PointBERT / ULIP_WITH_IMAGE), since PointBERT's CUDA
extensions (Chamfer/EMD/GPU KNN) are already working in this environment for
ULIP-2/OpenShape. In practice JM3D's PointBERT encoder needs none of those
extensions at inference time: grouping uses a pure-PyTorch knn_point() and a
stubbed knn_cuda.KNN (same stub already used by ULIP-2/OpenShape), and the
group-center FPS is JM3D's own pure-PyTorch iterative fps() (models/pointbert
/misc.py) -- the only broken bit was a *dead* top-level `from pointnet2_ops
import pointnet2_utils` import (unused, guarded by an already-commented-out
alternate fps() implementation), patched out below in-place.

Two more environment patches were required to make `models.ULIP_models`
importable at all (both in-place edits to the cloned JM3D repo, not to this
file):
  - utils/io.py: `import open3d` at module load time crashes with "Illegal
    instruction" on this cluster's login node (open3d's wheel uses AVX
    instructions the login node's virtual CPU lacks). We only ever use
    `IO._read_npy` (plain np.load) via the wildcard `from data.dataset_3d
    import *` chain, so `open3d`/`h5py` imports were made lazy (moved inside
    the _read_pcd/_read_h5 methods that actually need them).
  - data/dataset_3d.py: `from torch._six import string_classes` -- torch._six
    was removed in modern PyTorch. Replaced with `string_classes = str`
    (which is exactly what torch._six.string_classes was under Python 3).

JM3D's own data loaders (data/dataset_3d.py ModelNet/ShapeNet classes, with
ModelNet40.yaml's USE_NORMALS: FALSE) confirm the point encoder consumes xyz
only (no color channel) and use the same unit-sphere normalize_pc convention
(center + divide by max L2 norm) already used by this project's cached point
clouds -- so no renormalization is needed. Their loaders also default
self.uniform = True, i.e. farthest-point-sampling down to `npoints` (8192
for PointBERT) whenever the raw cloud has more points than that -- we
replicate this exactly in encode_pointclouds() using JM3D's own
models.pointbert.misc.fps() (pure PyTorch, GPU-capable) to downsample our
cached (10000, 6) clouds' xyz from 10000 -> 8192 points before encoding.
"""

import os
import sys
from pathlib import Path

JM3D_ROOT = '/msc/home/q45emuni/miccai_2026/JM3D'
if JM3D_ROOT not in sys.path:
    sys.path.insert(0, JM3D_ROOT)
# ULIP_PointBERT() loads its backbone config from a hardcoded path relative
# to the JM3D repo root ('./models/pointbert/...'), so cwd must match.
os.chdir(JM3D_ROOT)

CHECKPOINT_PATH = os.path.join(JM3D_ROOT, 'checkpoints', 'jm3d_pointbert.pt')
POINTCLOUD_ROOT = Path('/msc/home/q45emuni/foundation_models/shapemi/pointclouds')
NUM_POINTS = 8192  # JM3D PointBERT's default/trained point count.

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


class _DummyArgs:
    """Minimal args namespace satisfying ULIP_PointBERT()/PointTransformer's
    attribute reads. evaluate_3d=True skips JM3D's own internal loading of
    data/initialize_models/{point_bert_pretrained,slip_base_100ep}.pt (which
    we don't have and don't need -- we load the full trained JM3D checkpoint
    ourselves, overwriting all of these weights anyway). all_image=False
    selects the simpler ULIP_WITH_IMAGE architecture (vs. ULIP_WITH_ALL_IMAGE)
    -- we only need encode_text/encode_pc, not the multi-image fusion head or
    auxiliary classifier the training-time checkpoint also contains (those
    show up as harmless `unexpected_keys` when loading state_dict below).
    """
    evaluate_3d = True
    all_image = False
    class_text = 'None'


def build_model(device='cuda'):
    from models.ULIP_models import ULIP_PointBERT
    from utils.tokenizer import SimpleTokenizer

    model = ULIP_PointBERT(_DummyArgs())
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f'[jm3d] load_state_dict: missing={len(result.missing_keys)} '
              f'unexpected={len(result.unexpected_keys)}')
        if result.missing_keys:
            print('  missing[:5]:', result.missing_keys[:5])
        if result.unexpected_keys:
            print('  unexpected[:5]:', result.unexpected_keys[:5])
    model = model.to(device).eval()
    model.tokenizer = SimpleTokenizer()
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
    return torch.stack(text_features, dim=0)  # (num_classes, 512)


class PointCloudDataset(data.Dataset):
    def __init__(self, samples):
        """samples: list of (npy_path, label_idx)"""
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path).astype(np.float32)  # (10000, 6) xyzrgb
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
    from models.pointbert.misc import fps as jm3d_fps

    dataset = PointCloudDataset(samples)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    all_feats, all_labels, all_paths = [], [], []
    for pc, label, paths in loader:
        pc = pc.to(device, non_blocking=True)
        xyz = pc[:, :, :3].contiguous()  # JM3D's PointBERT consumes xyz only
        if xyz.shape[1] > NUM_POINTS:
            # Match JM3D's own data loader convention (self.uniform=True):
            # farthest-point-sample down to the model's trained point count.
            xyz = jm3d_fps(xyz, NUM_POINTS)
        feat = model.encode_pc(xyz)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_feats.append(feat.cpu())
        all_labels.append(label)
        all_paths.extend(paths)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0), all_paths
