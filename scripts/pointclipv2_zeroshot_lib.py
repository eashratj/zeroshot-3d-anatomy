"""
PointCLIP V2 zero-shot classification, adapted for MedShapeNet pathology
detection -- a genuinely independent third model relative to ULIP-2 and
OpenShape: it uses OpenAI's original CLIP checkpoint (not the OpenCLIP
ViT-bigG-14 / LAION-2B checkpoint ULIP-2 and OpenShape both share), and a
completely different mechanism -- render the point cloud to multiple 2D
depth-map "views", classify each view with ordinary 2D CLIP, and combine
across views. No point-cloud-specific pretraining is involved at all.

We bypass PointCLIP V2's Dassl3D training-framework scaffolding entirely
(trainers/__init__.py is empty, so importing trainers.mv_utils_zs directly
never touches dassl) and reuse only two pieces of their released code:
  - clip/clip.py: OpenAI's original CLIP loader + tokenizer
  - trainers/mv_utils_zs.py: Realistic_Projection, the multi-view renderer

Their released zero-shot pipeline also depends on `best_param.py`, a table
of per-dataset view-weights and prompt sets found via a hyperparameter
search specifically for ModelNet40/ScanObjectNN -- meaningless for our
kidney/brain categories. We use uniform view-weights (equivalent to a
simple average over views) instead, which is the standard, honest way to
run PointCLIP V2 zero-shot on a new dataset without dataset-specific
tuning.
"""

import sys
from pathlib import Path

POINTCLIPV2_ROOT = '/msc/home/q45emuni/miccai_2026/PointCLIP_V2/zeroshot_cls'
if POINTCLIPV2_ROOT not in sys.path:
    sys.path.insert(0, POINTCLIPV2_ROOT)

POINTCLOUD_ROOT = Path('/msc/home/q45emuni/foundation_models/shapemi/pointclouds')

TEMPLATES = [
    "a 3D model of a {}",
    "an anatomical shape of a {}",
    "a medical segmentation of a {}",
    "a point cloud of a {}",
    "a 3D scan of a {}",
]

BACKBONE = 'ViT-B/16'
IMSIZE = 224

import numpy as np
import torch
import torch.utils.data as data


def build_model(device='cuda'):
    from clip import clip
    from trainers.mv_utils_zs import Realistic_Projection

    clip_model = clip.load(BACKBONE, device=device)[0]
    clip_model = clip_model.float().eval()

    pc_views = Realistic_Projection()
    num_views = pc_views.num_views

    return {
        'clip_model': clip_model,
        'tokenizer': clip.tokenize,
        'get_img': pc_views.get_img,
        'num_views': num_views,
    }


@torch.no_grad()
def encode_class_text(model, class_names, device='cuda'):
    """Same ensemble-averaging convention as ULIP/OpenShape, using CLIP's
    own tokenizer/text encoder (OpenAI's, independent of OpenCLIP)."""
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
    return torch.stack(text_features, dim=0).float()  # (num_classes, D)


class PointCloudDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        pc = np.load(npy_path).astype(np.float32)[:, :3]  # xyz only -- geometry-only rendering
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
def encode_pointclouds(model, samples, batch_size=8, num_workers=4, device='cuda'):
    """Renders each shape to `num_views` 2D depth-map images, encodes each
    with CLIP's visual encoder, and averages the per-view (unit-normalized)
    image embeddings with uniform weights -- since no dataset-tuned
    per-view weights exist for our categories (see module docstring)."""
    clip_model = model['clip_model']
    get_img = model['get_img']
    num_views = model['num_views']

    dataset = PointCloudDataset(samples)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    all_feats, all_labels, all_paths = [], [], []
    for pc, label, paths in loader:
        pc = pc.to(device, non_blocking=True)
        images = get_img(pc)  # (B*num_views, 3, H, W)
        images = torch.nn.functional.interpolate(
            images, size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)
        image_feat = clip_model.encode_image(images.float())
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        B = pc.shape[0]
        image_feat = image_feat.reshape(B, num_views, -1).mean(dim=1)  # uniform average over views
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        all_feats.append(image_feat.cpu())
        all_labels.append(label)
        all_paths.extend(paths)
    return torch.cat(all_feats, dim=0).float(), torch.cat(all_labels, dim=0), all_paths
