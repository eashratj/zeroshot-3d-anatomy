"""
Uni3D (Zhou et al., ICLR'24 Spotlight) zero-shot classification, adapted
for MedShapeNet -- a 5th, independent zero-shot model added to test
whether the "embedding separates classes fine, zero-shot text mapping is
the bottleneck" finding generalizes further. Uni3D is architecturally
distinct from all 4 existing models: a native 3D ViT point-cloud encoder
(like ULIP-2/OpenShape) but aligned to a *third* independent text tower
-- EVA02-E-14-plus (EVA-CLIP), trained on LAION-2B, distinct from both
OpenCLIP ViT-bigG-14/LAION-2B (ULIP-2, OpenShape) and OpenAI's original
CLIP (PointCLIP V2, DiffCLIP's renderer pairing).

Reimplemented standalone (bypassing Uni3D's main.py/Dassl-like training
scaffolding), same pattern used for PointCLIP V2. Uni3D's own point
tokenizer depends on the `pointnet2_ops` CUDA extension for farthest
point sampling, which is unavailable on this cluster's torch/cuda build
-- patched in Uni3D/models/point_encoder.py with the same pure-PyTorch
iterative FPS fallback already used for ULIP/OpenShape.

Uni3D's own dataset loader (data/datasets.py) confirms both preprocessing
conventions already used by this project's pipeline: unit-sphere xyz
normalization (center + divide by max L2 norm) and constant gray
rgb=0.4 fill for colorless meshes ("scale to 0.4 to make it consistent
with the training data") -- so our existing cached .npy point clouds
need no reformatting.
"""

import sys
from pathlib import Path

UNI3D_ROOT = '/msc/home/q45emuni/miccai_2026/Uni3D'
CLIP_CKPT_PATH = '/msc/home/q45emuni/miccai_2026/Uni3D/clip_model/eva02_e14_plus.bin'
UNI3D_CKPT_PATH = '/msc/home/q45emuni/miccai_2026/Uni3D/checkpoints/uni3d-b.pt'
CLIP_MODEL_NAME = 'EVA02-E-14-plus'

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
import torch.nn as nn
import torch.utils.data as data
from easydict import EasyDict


def build_model(device='cuda'):
    if UNI3D_ROOT not in sys.path:
        sys.path.insert(0, UNI3D_ROOT)
    import timm
    import open_clip
    # Import the encoder module directly (not `models.uni3d`, which pulls
    # in `models/losses.py` -> `utils/utils.py` -> `data/datasets.py` and
    # its own dataset-scaffolding dependencies we don't need here).
    from models.point_encoder import PointcloudEncoder

    args = EasyDict(
        pc_model='eva02_base_patch14_448',
        pretrained_pc='',
        drop_path_rate=0.0,
        pc_feat_dim=768,
        embed_dim=1024,
        group_size=64,
        num_group=512,
        pc_encoder_dim=512,
        patch_dropout=0.0,
    )
    point_transformer = timm.create_model(
        args.pc_model, checkpoint_path=args.pretrained_pc, drop_path_rate=args.drop_path_rate)
    point_encoder = PointcloudEncoder(point_transformer, args)

    class Uni3DModel(nn.Module):
        def __init__(self, point_encoder):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.point_encoder = point_encoder

        def encode_pc(self, pc):
            xyz = pc[:, :, :3].contiguous()
            color = pc[:, :, 3:].contiguous()
            return self.point_encoder(xyz, color)

    model = Uni3DModel(point_encoder)
    ckpt = torch.load(UNI3D_CKPT_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['module'] if 'module' in ckpt else ckpt
    sd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[uni3d] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('  missing[:5]:', missing[:5])
        if unexpected:
            print('  unexpected[:5]:', unexpected[:5])
    model = model.to(device).eval()

    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=CLIP_MODEL_NAME, pretrained=CLIP_CKPT_PATH)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    return {'model': model, 'clip_model': clip_model, 'tokenizer': tokenizer}


@torch.no_grad()
def encode_class_text(model, class_names, device='cuda'):
    clip_model = model['clip_model']
    tokenizer = model['tokenizer']
    text_features = []
    for name in class_names:
        texts = [t.format(name) for t in TEMPLATES]
        tokens = tokenizer(texts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        emb = emb / emb.norm(dim=0)
        text_features.append(emb)
    return torch.stack(text_features, dim=0).float()


class PointCloudDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path).astype(np.float32)  # (10000, 6) xyz+rgb, already unit-sphere normalized
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
def encode_pointclouds(model, samples, batch_size=32, num_workers=4, device='cuda'):
    net = model['model']
    dataset = PointCloudDataset(samples)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    all_feats, all_labels, all_paths = [], [], []
    for pc, label, paths in loader:
        pc = pc.to(device, non_blocking=True)
        feat = net.encode_pc(pc)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_feats.append(feat.cpu())
        all_labels.append(label)
        all_paths.extend(paths)
    return torch.cat(all_feats, dim=0).float(), torch.cat(all_labels, dim=0), all_paths
