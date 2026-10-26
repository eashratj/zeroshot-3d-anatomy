"""
Shared utilities for Point-Bind zero-shot classification on MedShapeNet point
clouds. Mirrors ulip_zeroshot_lib.py's interface.

Point-Bind (Guo et al., arXiv:2309.00615) aligns a 3D point-cloud encoder into
the joint embedding space of ImageBind (image/text/audio/video/IMU), giving a
6th independent zero-shot text-image-point alignment (distinct text/vision
tower from all of ULIP-2/OpenShape's OpenCLIP ViT-bigG-14, Uni3D's EVA-CLIP,
and PointCLIP V2's OpenAI CLIP -- this one is ImageBind's own text/vision
towers, contrastively trained by Meta on web image-text-audio-video-depth-IMU
pairs, not LAION).

Encoder variant: Point-BERT, NOT the paper's better-performing I2P-MAE variant
(76.3% vs 78.0% ModelNet40 zero-shot in the Point-Bind paper). Chosen because
it is the simpler, more robust loading path on this cluster:
  - Both variants' Point-Bind checkpoints are full self-contained state dicts
    (point encoder + projection + the *entire* ImageBind-Huge backbone, since
    imagebind_huge() has no pretrained=True auto-download in this vendored
    copy -- see models/PointBind_models.py). Both download fine from Google
    Drive via gdown (see Install.md).
  - Both variants' upstream code has a dead `from pointnet2_ops import
    pointnet2_utils` import (CUDA extension, unavailable on this cluster).
    For Point-BERT (models/pointbert/misc.py) this import is *unused* --
    fps() there is already a pure-PyTorch iterative FPS -- so the fix is a
    one-line deletion. For I2P-MAE (models/i2p_mae/misc.py) fps() genuinely
    calls pointnet2_utils.furthest_point_sample, requiring a real patch
    (pure-PyTorch FPS substituted, see that file) to even use it.
  - Point-BERT's dvae.py Group (the module actually used for FPS+grouping)
    already uses a pure-PyTorch knn_point (torch.topk) instead of knn_cuda;
    I2P-MAE's Group class genuinely instantiates knn_cuda.KNN() and depends
    on it at forward time. knn_cuda itself *is* installed in this env (same
    package ULIP already depends on), so this isn't blocking either variant,
    but it's one less moving part for Point-BERT.
  - Point-Bind's own README lists Point-BERT first as the more standard
    config (PointTransformer_8192point.yaml, 8192 pts, matching the familiar
    ULIP/Point-BERT lineage already used elsewhere in this project).
Net effect: Point-BERT needed only a dead-import deletion; I2P-MAE needed a
real algorithmic patch to even import. Point-BERT chosen as the lower-risk
path given the same-day deadline. (i2p_mae/misc.py was still patched for
import-safety, since PointBind_models.py transitively imports it regardless
of which variant is instantiated -- but its patched fps() is never exercised
by the Point-BERT path used here.)

The upstream `models.PointBind_models` module is deliberately NOT imported
here: `from data.dataset_3d import *` pulls in `utils/io.py` -> `import
open3d`, and this cluster's open3d build SIGILLs on import on the (GPU-less)
login node -- and crashes are plausible on some compute nodes too, so it's
avoided outright. None of that dataset-loading machinery is actually used by
PointBind_models.py itself (dead wildcard import), so this file reimplements
the small POINTBIND wrapper directly from imagebind + the pointbert encoder,
the same "import the encoder module directly, skip the training scaffolding"
approach already used in uni3d_zeroshot_lib.py.

Point-Bind's Point-BERT encoder is xyz-only (unlike ULIP-2/OpenShape's
colored variant) -- see models/pointbert/dvae.py's Encoder/Group, which
hardcode 3 input channels with no RGB branch. encode_pointclouds therefore
drops the constant-gray RGB columns from the cached (10000, 6) arrays before
encoding.
"""

import os
import sys
from pathlib import Path

POINTBIND_ROOT = '/msc/home/q45emuni/miccai_2026/Point-Bind'
if POINTBIND_ROOT not in sys.path:
    sys.path.insert(0, POINTBIND_ROOT)
# cfg_from_yaml_file's caller / bpe vocab path are relative to the repo root.
os.chdir(POINTBIND_ROOT)

CHECKPOINT_PATHS = {
    'pointbert': os.path.join(POINTBIND_ROOT, 'ckpts', 'pointbind_pointbert.pt'),
    'i2pmae': os.path.join(POINTBIND_ROOT, 'ckpts', 'pointbind_i2pmae.pt'),
}
# 2026-10-16: switched default to i2pmae -- the better-performing variant in
# the Point-Bind paper (78.0% vs 76.3% ModelNet40 zero-shot). Point-BERT was
# only used earlier because of a same-day deadline; the actual blocker
# (I2P-MAE's fps() needing pointnet2_ops, a CUDA extension unavailable on
# this cluster) was already patched to pure-PyTorch FPS in
# models/i2p_mae/misc.py by that same session, and knn_cuda (I2P-MAE's
# Group class dependency) was already installed -- so nothing was actually
# stopping this, it just hadn't been switched over yet.
CHECKPOINT_PATH = CHECKPOINT_PATHS['i2pmae']
POINTBERT_CONFIG = os.path.join(POINTBIND_ROOT, 'models', 'pointbert', 'PointTransformer_8192point.yaml')
BPE_PATH = os.path.join(POINTBIND_ROOT, 'bpe', 'bpe_simple_vocab_16e6.txt.gz')
POINTCLOUD_ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi/pointclouds')

TEMPLATES = [
    "a 3D model of a {}",
    "an anatomical shape of a {}",
    "a medical segmentation of a {}",
    "a point cloud of a {}",
    "a 3D scan of a {}",
]

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


def build_model(device='cuda', variant='i2pmae'):
    from imagebind import imagebind_model
    from imagebind.imagebind_model import ModalityType
    from imagebind.multimodal_preprocessors import SimpleTokenizer

    class DummyArgs:
        pass

    class POINTBIND(nn.Module):
        """Reimplementation of models.PointBind_models.POINTBIND, trimmed to
        avoid the data.dataset_3d -> utils.io -> open3d import chain (see
        module docstring). Field names match upstream exactly so either
        checkpoint's state dict loads directly."""

        def __init__(self, point_encoder, pc_feat_dims):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.bind = imagebind_model.imagebind_huge()
            self.point_encoder = point_encoder
            self.pc_projection = nn.Parameter(torch.empty(pc_feat_dims, 512))
            nn.init.normal_(self.pc_projection, std=512 ** -0.5)

        def encode_pc(self, pc):
            pc_feat = self.point_encoder(pc)
            pc_embed = pc_feat @ self.pc_projection
            pc_embed = self.bind.modality_head_point(pc_embed)
            pc_embed = self.bind.modality_postprocessor_point(pc_embed)
            return pc_embed

    if variant == 'i2pmae':
        from models.i2p_mae.I2P_MAE import I2P_MAE_BIND
        point_encoder = I2P_MAE_BIND()
        pc_feat_dims = 384  # see models/PointBind_models.py: PointBind_I2PMAE
    elif variant == 'pointbert':
        from utils.config import cfg_from_yaml_file
        from models.pointbert.point_encoder import PointTransformer_BIND
        config = cfg_from_yaml_file(POINTBERT_CONFIG)
        point_encoder = PointTransformer_BIND(config.model, args=DummyArgs())
        pc_feat_dims = 768  # 2 * trans_dim(384), see PointTransformer_BIND.forward
    else:
        raise ValueError(f'Unknown variant {variant!r}')

    model = POINTBIND(point_encoder, pc_feat_dims=pc_feat_dims)

    state_dict = torch.load(CHECKPOINT_PATHS[variant], map_location='cpu', weights_only=False)
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f'[pointbind] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('  missing[:5]:', missing[:5])
        if unexpected:
            print('  unexpected[:5]:', unexpected[:5])
    model = model.to(device).eval()

    model.tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    model.ModalityType = ModalityType
    return model


@torch.no_grad()
def encode_class_text(model, class_names, device='cuda'):
    """Ensemble-averaged, L2-normalized text embedding per class name."""
    tokenizer = model.tokenizer
    ModalityType = model.ModalityType
    text_features = []
    for name in class_names:
        texts = [t.format(name) for t in TEMPLATES]
        tokens = torch.cat([tokenizer(t).unsqueeze(0) for t in texts], dim=0).to(device)
        embeddings = model.bind({ModalityType.TEXT: tokens})
        emb = embeddings[ModalityType.TEXT]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_features.append(emb)
    return torch.stack(text_features, dim=0)  # (num_classes, 1024)


class PointCloudDataset(data.Dataset):
    def __init__(self, samples):
        """samples: list of (npy_path, label_idx)"""
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path).astype(np.float32)  # (10000, 6) xyz + constant gray rgb
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
        xyz = pc[:, :, :3].contiguous()  # Point-Bind's Point-BERT encoder is xyz-only, no RGB branch
        feat = model.encode_pc(xyz)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_feats.append(feat.cpu())
        all_labels.append(label)
        all_paths.extend(paths)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0), all_paths
