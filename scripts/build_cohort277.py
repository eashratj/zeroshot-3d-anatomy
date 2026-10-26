"""
Download + preprocess the MedShapeNet 277-subject cohort for the ISBI
granularity-gradient tasks (T1-T5). See ISBI_EXPERIMENT_SPEC.md.

For each category:
  1. filter the master index to  <cohort_sid>_<category>.stl  (277 files)
  2. download STLs in parallel to a temp dir
  3. sample 10,000 surface points, normalize (centroid-center, unit-sphere)
     -- IDENTICAL to scripts/preprocess.py::pc_normalize
  4. save float32 (10000, 6) .npy  [xyz + constant gray 0.4 rgb]
  5. delete the STLs for that category, move on

Resumable: skips any .npy that already exists; a category is "done" when
all its cohort .npy files are present.

Output: <POINTCLOUD_ROOT>/<category>/<sid>_<category>.npy
        POINTCLOUD_ROOT = /msc/home/q45emuni/foundation_models/shapemi/pointclouds
        (the path scripts/ulip_zeroshot_lib.py etc. already expect)

Usage:
  python build_cohort277.py                      # all categories in the list file
  python build_cohort277.py --categories vertebraeL3 heartatriumleft
  python build_cohort277.py --workers 8
"""

import argparse
import gc
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import trimesh

INDEX_URL = 'https://mml.ikim.nrw/ait/projects/medshapenet/MedShapeNetDataset.txt'
# NOTE 2026-09-14: previously pointed at a session-scoped /tmp scratchpad
# that no longer exists once that session ended -- moved to a durable
# project-level location so this survives across sessions. cohort_sids.txt
# was regenerated and verified this session (277 subjects, confirmed via
# 7 categories with IDENTICAL subject-ID sets: liver/heart/colon/trachea/
# esophagus/duodenum/bladder, all exactly 277). download_categories.txt
# covers all of T1-T6 (75 unique categories), each name verified to exist
# in the live index before being added -- see tasks_categories.txt in the
# same dir for the per-task breakdown.
SCRATCH = Path('/msc/home/q45emuni/miccai_2026/shapemi/cohort277_meta')
COHORT_SIDS_FILE = SCRATCH / 'cohort_sids.txt'
CATEGORIES_FILE = SCRATCH / 'download_categories.txt'
INDEX_CACHE = SCRATCH / 'MedShapeNetDataset_full.txt'

POINTCLOUD_ROOT = Path('/msc/home/q45emuni/foundation_models/shapemi/pointclouds')
STL_TMP = Path('/msc/home/q45emuni/Data/medshapenet/_cohort277_stl_tmp')
LOG = Path('/msc/home/q45emuni/miccai_2026/shapemi/cohort277_build.log')

N_POINTS = 10000
SEED = 0
N_WORKERS = 6


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m


def get_index_lines():
    if INDEX_CACHE.exists():
        return [l.strip() for l in INDEX_CACHE.read_text().splitlines() if l.strip()]
    log(f'fetching index from {INDEX_URL}')
    r = requests.get(INDEX_URL, timeout=60)
    r.raise_for_status()
    INDEX_CACHE.write_text(r.text)
    return [l.strip() for l in r.text.splitlines() if l.strip()]


def fname(u):
    return u.split('/')[-1]


def download_one(url, dst, retries=3):
    # 2026-09-14 fix: the sciebo host intermittently drops the connection
    # mid-stream (503s and read-timeouts observed live in this run's log)
    # WITHOUT requests raising -- iter_content just ends early and the
    # partial file gets renamed as if it were complete. That silently
    # truncated/corrupted STL then fails trimesh's binary-length check and
    # falls through to the ASCII parser, which is what actually produced
    # the "`endsolid` precedes `solid`!" errors seen across many
    # categories (not a trimesh detection bug -- confirmed by reading
    # trimesh.exchange.stl.load_stl_binary's own size-formula check).
    # Fix: verify Content-Length against bytes actually written, and retry
    # with backoff on any failure (HTTP error, timeout, or length mismatch)
    # instead of accepting a short file as success.
    if dst.exists():
        return True
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            expected_len = r.headers.get('Content-Length')
            expected_len = int(expected_len) if expected_len is not None else None
            tmp = dst.with_suffix('.part')
            n_written = 0
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n_written += len(chunk)
            if expected_len is not None and n_written != expected_len:
                raise IOError(f'truncated download: got {n_written}B, expected {expected_len}B')
            tmp.rename(dst)
            return True
        except Exception as e:
            if dst.with_suffix('.part').exists():
                dst.with_suffix('.part').unlink()
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                log(f'  DOWNLOAD FAIL {dst.name}: {e} (after {retries} attempts)')
    return False


def process_one(stl_path, npy_path):
    mesh = None
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        rng = np.random.default_rng(SEED)
        pts, _ = trimesh.sample.sample_surface(mesh, N_POINTS, seed=rng)
        pts = pc_normalize(np.asarray(pts, dtype=np.float32))
        rgb = np.ones_like(pts) * 0.4
        pc6 = np.concatenate([pts, rgb], axis=1).astype(np.float32)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = npy_path.with_suffix('.tmp.npy')
        np.save(tmp, pc6)
        tmp.rename(npy_path)
        return True
    except Exception as e:
        log(f'  PROCESS FAIL {stl_path.name}: {e}')
        return False
    finally:
        # STL meshes for the big categories (colon, lungs, hips...) decode to
        # several hundred MB each; with several running concurrently this
        # blew the login node's (invisible, no ulimit/cgroup showing it)
        # session memory cap and got SIGKILL'd. Force release promptly
        # rather than waiting on GC scheduling across worker threads.
        del mesh
        gc.collect()


def do_category(cat, index_lines, cohort_sids, workers, proc_workers):
    pat = re.compile(r'^(\d{6})_%s\.stl$' % re.escape(cat))
    urls = []
    for u in index_lines:
        m = pat.match(fname(u))
        if m and m.group(1) in cohort_sids:
            urls.append(u)
    out_dir = POINTCLOUD_ROOT / cat
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [u for u in urls if not (out_dir / fname(u).replace('.stl', '.npy')).exists()]
    if not todo:
        log(f'{cat}: all {len(urls)} already done, skip')
        return
    log(f'{cat}: {len(urls)} cohort files, {len(todo)} to build')

    stl_dir = STL_TMP / cat
    stl_dir.mkdir(parents=True, exist_ok=True)

    # download
    t0 = time.time()
    ok_dl = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, u, stl_dir / fname(u)): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok_dl += 1
            if i % 50 == 0:
                log(f'  {cat}: downloaded {i}/{len(todo)} ({(time.time()-t0)/60:.1f} min)')
    log(f'  {cat}: download done {ok_dl}/{len(todo)} in {(time.time()-t0)/60:.1f} min')

    # process (memory-bound, not I/O-bound -- deliberately fewer workers
    # than the download phase; see process_one's docstring comment)
    t0 = time.time()
    ok_pr = 0
    with ThreadPoolExecutor(max_workers=proc_workers) as ex:
        futs = {}
        for u in todo:
            s = stl_dir / fname(u)
            if s.exists():
                futs[ex.submit(process_one, s, out_dir / fname(u).replace('.stl', '.npy'))] = s
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok_pr += 1
            if i % 50 == 0:
                log(f'  {cat}: processed {i}/{len(futs)} ({(time.time()-t0)/60:.1f} min)')
    log(f'  {cat}: process done {ok_pr}/{len(todo)} in {(time.time()-t0)/60:.1f} min')

    # cleanup STLs for this category
    shutil.rmtree(stl_dir, ignore_errors=True)
    log(f'  {cat}: STLs deleted. npy count = {len(list(out_dir.glob("*.npy")))}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=None)
    ap.add_argument('--workers', type=int, default=N_WORKERS, help='download concurrency (I/O-bound)')
    ap.add_argument('--proc-workers', type=int, default=2, help='mesh-processing concurrency (memory-bound)')
    args = ap.parse_args()

    cohort_sids = set(l.strip() for l in COHORT_SIDS_FILE.read_text().splitlines() if l.strip())
    cats = args.categories or [c.strip() for c in CATEGORIES_FILE.read_text().splitlines() if c.strip()]
    log(f'=== build_cohort277 start: {len(cats)} categories, {len(cohort_sids)} subjects, '
        f'{args.workers} dl-workers, {args.proc_workers} proc-workers ===')
    index_lines = get_index_lines()

    for j, cat in enumerate(cats, 1):
        log(f'--- [{j}/{len(cats)}] {cat} ---')
        try:
            do_category(cat, index_lines, cohort_sids, args.workers, args.proc_workers)
        except Exception as e:
            log(f'{cat}: CATEGORY FAILED: {e}')

    shutil.rmtree(STL_TMP, ignore_errors=True)
    log('=== build_cohort277 done ===')


if __name__ == '__main__':
    main()
