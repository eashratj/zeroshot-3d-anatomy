"""
Re-run the SHAPEMI pilot's segmentation-protocol confound diagnostic on
the current (277-cohort) T6 data: does normal-vs-tumor mesh fragmentation
(disconnected component count) and convex-hull volume show the same
suspicious backwards pattern for brain that suggested a pipeline artifact
rather than real pathology signal? Normal and tumor shapes are
necessarily cross-source (different patients, different studies, likely
different segmentation pipelines) -- this checks whether that shows up
as a confound in the raw geometry.

60 raw meshes per category, matching the original pilot's sample size.
Needs raw STLs: tumor-side ones already exist locally
(/msc/home/q45emuni/Data/medshapenet/tumoredKidney|tumoredbrain/); normal-
side ones are re-downloaded fresh since build_cohort277.py deletes raw
STLs after processing into point clouds.

Usage: python check_pathology_confound.py
"""

import random
import re
from pathlib import Path

import numpy as np
import requests
import trimesh

INDEX_CACHE = Path('/msc/home/q45emuni/miccai_2026/shapemi/cohort277_meta/MedShapeNetDataset_full.txt')
COHORT_SIDS_FILE = Path('/msc/home/q45emuni/miccai_2026/shapemi/cohort277_meta/cohort_sids.txt')
TMP_DIR = Path('/tmp/confound_check_stl')
N_SAMPLE = 60
SEED = 0

TUMOR_STL_DIRS = {
    'kidney': Path('/msc/home/q45emuni/Data/medshapenet/tumoredKidney'),
    'brain': Path('/msc/home/q45emuni/Data/medshapenet/tumoredbrain'),
}
NORMAL_CATEGORY = {'kidney': 'kidneyleft', 'brain': 'brain'}


def get_index_lines():
    return [l.strip() for l in INDEX_CACHE.read_text().splitlines() if l.strip()]


def fetch_normal_stls(organ, n):
    cat = NORMAL_CATEGORY[organ]
    cohort_sids = [l.strip() for l in COHORT_SIDS_FILE.read_text().splitlines() if l.strip()]
    rng = random.Random(SEED)
    sample_sids = rng.sample(cohort_sids, n)
    pat = re.compile(rf'^({"|".join(sample_sids)})_{cat}\.stl$')
    index_lines = get_index_lines()
    urls = {}
    for u in index_lines:
        fname = u.split('/')[-1]
        m = pat.match(fname)
        if m:
            urls[m.group(1)] = u

    out_dir = TMP_DIR / f'normal_{organ}'
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sid, url in urls.items():
        dst = out_dir / f'{sid}_{cat}.stl'
        if not dst.exists():
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                dst.write_bytes(r.content)
            except Exception as e:
                print(f'  download fail {sid}: {e}')
                continue
        paths.append(dst)
    return paths


def mesh_stats(stl_path):
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        n_components = len(mesh.split(only_watertight=False))
        hull_volume = mesh.convex_hull.volume
        return n_components, hull_volume
    except Exception as e:
        return None, None


def summarize(label, paths):
    comps, vols = [], []
    for p in paths:
        n, v = mesh_stats(p)
        if n is not None:
            comps.append(n)
            vols.append(v)
    comps = np.array(comps)
    vols = np.array(vols)
    print(f'{label}: n={len(comps)}  mean_components={comps.mean():.1f} '
          f'(median {np.median(comps):.0f})  mean_hull_vol={vols.mean():.1f}')
    return comps, vols


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    for organ in ['kidney', 'brain']:
        print(f'\n=== {organ} ===')
        tumor_paths = list(TUMOR_STL_DIRS[organ].glob('*.stl'))
        tumor_sample = rng.sample(tumor_paths, min(N_SAMPLE, len(tumor_paths)))
        normal_paths = fetch_normal_stls(organ, N_SAMPLE)

        n_comp, n_vol = summarize('normal', normal_paths)
        t_comp, t_vol = summarize('tumor ', tumor_sample)

        ratio = t_comp.mean() / n_comp.mean() if n_comp.mean() > 0 else float('nan')
        vol_ratio = t_vol.mean() / n_vol.mean() if n_vol.mean() > 0 else float('nan')
        direction = 'MORE fragmented (as pilot found for kidney)' if ratio > 1 else 'FEWER components (as pilot found for brain -- suspicious)'
        print(f'  tumor/normal component ratio: {ratio:.2f}x -- {direction}')
        print(f'  tumor/normal hull-volume ratio: {vol_ratio:.2f}x')


if __name__ == '__main__':
    main()
