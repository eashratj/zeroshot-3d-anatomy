"""
Robustness corruption pipeline: bridges clean MedShapeNet meshes toward
realistic intraoperative 3D-reconstruction conditions. See
ISBI_EXPERIMENT_SPEC.md Sec. 6.

Three axes, each corrupting the *point cloud* (post-preprocessing, i.e.
the cached (N, 6) xyz+rgb .npy arrays already centered/unit-sphere-scaled
-- NOT the raw mesh), applied once per (source file, axis, severity) with
a fixed seed so every model sees the identical corrupted shape for a
given condition -- essential for a fair cross-model comparison. Severity
levels are taken verbatim from the spec table, which is NOT uniformly 4
levels per axis (downsampling has 5, not 4 -- see NOTE below).

  Partiality (4 levels): keep fraction p of points via a random
    spherical-cap crop (pick a random unit direction, keep the p-fraction
    of points closest to that direction from the centroid) -- simulates a
    single-viewpoint intraoperative capture seeing only part of a surface.
  Noise (4 levels): additive per-point Gaussian jitter, sigma in
    unit-sphere coordinate units -- simulates 3D-reconstruction noise.
  Downsampling (5 levels): uniform random subsample to N points --
    simulates a sparse reconstruction. N=10000 (severity 0) is a no-op,
    identical to the uncorrupted input.

NOTE on run count: the ISBI_EXPERIMENT_SPEC.md decision log and this
session's run-count arithmetic (1,296 = 6 tasks x 6 models x 3 axes x
4 levels x 3 probes) assumed a uniform 4 severity levels per axis. The
spec's own Sec. 6 table lists 4/4/5 for partiality/noise/downsampling
respectively. Using the real per-axis counts: 6 tasks x 6 models x
(4+4+5) axis-severity combinations x 3 probes = 1,404 runs (108 more
than the 1,296 figure already agreed) -- OR, if the "no corruption"
condition (p=1.0 / sigma=0 / N=10000) is shared across all 3 axes rather
than tripled, 6 x 6 x (3+3+4 + 1 shared clean) x 3 = 1,224. Flagging this
now rather than silently picking one -- confirm which convention before
the Slurm array job is sized off of it.

Usage:
  python robustness_corrupt.py --task kidney --src-root /path/to/clean/pointclouds
  python robustness_corrupt.py --list-conditions   # just print the axis/severity grid
"""

import argparse
from pathlib import Path

import numpy as np

OUT_ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi/pointclouds_robust')

PARTIALITY_LEVELS = [1.0, 0.75, 0.5, 0.25]      # keep-fraction
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.04]          # per-point sigma, unit-sphere units
DOWNSAMPLE_LEVELS = [10000, 4096, 2048, 1024, 512]  # target point count

AXES = {
    'partiality': PARTIALITY_LEVELS,
    'noise': NOISE_LEVELS,
    'downsample': DOWNSAMPLE_LEVELS,
}


def _seed_for(path: Path, axis: str, severity) -> int:
    """Deterministic per-(file, axis, severity) seed so re-running is
    reproducible and every model consuming this condition sees the exact
    same corrupted point cloud."""
    key = f'{path.stem}|{axis}|{severity}'
    return abs(hash(key)) % (2**32 - 1)


def corrupt_partiality(pc: np.ndarray, keep_frac: float, seed: int) -> np.ndarray:
    """Random spherical-cap crop: keep the keep_frac of points closest
    (by dot product) to a random unit direction from the centroid --
    simulates a single-viewpoint capture. keep_frac=1.0 is a no-op."""
    if keep_frac >= 1.0:
        return pc
    rng = np.random.default_rng(seed)
    xyz = pc[:, :3]
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    scores = xyz @ direction
    n_keep = max(int(round(len(pc) * keep_frac)), 1)
    keep_idx = np.argsort(-scores)[:n_keep]
    return pc[keep_idx]


def corrupt_noise(pc: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Additive Gaussian jitter on xyz only (rgb channel untouched).
    sigma=0.0 is a no-op."""
    if sigma <= 0.0:
        return pc
    rng = np.random.default_rng(seed)
    out = pc.copy()
    out[:, :3] = out[:, :3] + rng.normal(scale=sigma, size=out[:, :3].shape)
    return out


def corrupt_downsample(pc: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Uniform random subsample to n_target points. n_target >= len(pc)
    is a no-op (never upsamples)."""
    if n_target >= len(pc):
        return pc
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pc), size=n_target, replace=False)
    return pc[idx]


CORRUPT_FN = {
    'partiality': corrupt_partiality,
    'noise': corrupt_noise,
    'downsample': corrupt_downsample,
}


def process_file(src: Path, dst: Path, axis: str, severity):
    if dst.exists():
        return
    pc = np.load(src)
    seed = _seed_for(src, axis, severity)
    out = CORRUPT_FN[axis](pc, severity, seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.tmp.npy')
    np.save(tmp, out.astype(np.float32))
    tmp.rename(dst)


def run_condition(src_root: Path, task: str, axis: str, severity):
    """Corrupt every .npy under src_root for one (task, axis, severity)
    condition, mirroring the category subdirectory structure into
    OUT_ROOT/<task>/<axis>_<severity>/<category>/<file>.npy."""
    out_dir = OUT_ROOT / task / f'{axis}_{severity}'
    files = sorted(src_root.rglob('*.npy'))
    for f in files:
        rel = f.relative_to(src_root)
        dst = out_dir / rel
        process_file(f, dst, axis, severity)
    return len(files), out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=False)
    ap.add_argument('--src-root', required=False,
                     help='Directory of clean, preprocessed .npy point clouds for this task.')
    ap.add_argument('--list-conditions', action='store_true')
    args = ap.parse_args()

    if args.list_conditions:
        total = 0
        for axis, levels in AXES.items():
            print(f'{axis}: {levels}  ({len(levels)} levels)')
            total += len(levels)
        print(f'Total axis-severity combinations per task per model: {total}')
        return

    if not args.task or not args.src_root:
        ap.error('--task and --src-root required unless --list-conditions')

    src_root = Path(args.src_root)
    for axis, levels in AXES.items():
        for sev in levels:
            n, out_dir = run_condition(src_root, args.task, axis, sev)
            print(f'{args.task} / {axis}={sev}: {n} files -> {out_dir}')


if __name__ == '__main__':
    main()
