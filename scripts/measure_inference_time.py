"""
Measure per-query inference time for each of the five scoring methods used
in the paper, isolating the classifier/probe cost from feature extraction
(identical across all methods, so not the differentiator here). Uses
already-cached embeddings from feature_cache/ -- no GPU/re-encoding needed.

Methods measured:
  - zero-shot: one matmul against fixed text embeddings.
  - kNN (as used in the paper's probe): scikit-learn KNeighborsClassifier,
    fit on ~80% of the cohort (matching the paper's 5-fold CV), timed on
    the predict() call for the held-out fold.
  - clustering (k-means probe): there is no separate "fit once, predict
    many" step for a NEW query under this paradigm -- unsupervised k-means
    only forms stable, Hungarian-matchable clusters by seeing the whole
    set at once, so "inference" on one new point means re-running
    fit_predict on the whole dataset. Timed as such.
  - cache-adapter: affinity against the tiny K-shot cache only.
  - soft-prompt tuning: identical cost to zero-shot once trained (a fixed,
    already-learned set of class text embeddings) -- the K-shot training
    cost is separate and one-time, not paid per query.

Usage:
  python measure_inference_time.py
"""

import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from run_robustness_condition import FEATURE_CACHE_ROOT

TASK = 'T5_vertebrae'
MODEL = 'ulip2'
N_REPEATS = 20


def load_features():
    d = np.load(FEATURE_CACHE_ROOT / f'{TASK}__{MODEL}__clean.npz', allow_pickle=True)
    return d['feats'].astype(np.float32), d['gt']


def time_zero_shot(feats, gt, n_classes):
    text_features = np.random.randn(n_classes, feats.shape[1]).astype(np.float32)
    text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
    query = feats[:1]

    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        logits = query @ text_features.T
        pred = logits.argmax(axis=1)
    dt = (time.perf_counter() - t0) / N_REPEATS
    return dt


def time_soft_prompt(feats, gt, n_classes):
    # Identical cost to zero-shot at inference -- the learned text
    # embeddings are just a fixed (n_classes, dim) matrix by this point.
    return time_zero_shot(feats, gt, n_classes)


def time_cache_adapter(feats, gt, n_classes, k=16):
    rng = np.random.RandomState(0)
    support_idx = []
    for c in range(n_classes):
        class_idx = np.where(gt == c)[0]
        support_idx.extend(rng.choice(class_idx, size=min(k, len(class_idx)), replace=False))
    support_idx = np.array(support_idx)
    keys = feats[support_idx]
    query = feats[:1]

    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        cos_sim = query @ keys.T
        affinity = np.exp(-5.5 * (1 - cos_sim))
        cache_logits = affinity @ np.eye(n_classes)[gt[support_idx]]
    dt = (time.perf_counter() - t0) / N_REPEATS
    return dt, len(support_idx)


def time_knn(feats, gt, n_classes):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    train_idx, test_idx = next(skf.split(feats, gt))
    scaler = StandardScaler().fit(feats[train_idx])
    clf = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(feats[train_idx]), gt[train_idx])
    query = scaler.transform(feats[test_idx[:1]])

    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        pred = clf.predict(query)
    dt = (time.perf_counter() - t0) / N_REPEATS
    return dt, len(train_idx)


def time_clustering(feats, gt, n_classes):
    feats_s = StandardScaler().fit_transform(feats)

    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
    km.fit_predict(feats_s)
    dt = time.perf_counter() - t0
    return dt, len(feats)


def main():
    feats, gt = load_features()
    n_classes = len(np.unique(gt))
    print(f'Task: {TASK}, model: {MODEL}, n_samples={len(feats)}, n_classes={n_classes}')
    print()

    zs_time = time_zero_shot(feats, gt, n_classes)
    print(f'Zero-shot:        {zs_time*1000:.4f} ms/query (fixed cost, O(n_classes))')

    sp_time = time_soft_prompt(feats, gt, n_classes)
    print(f'Soft prompt:      {sp_time*1000:.4f} ms/query (same as zero-shot at inference; '
          f'one-time training cost paid separately, not per query)')

    ca_time, cache_size = time_cache_adapter(feats, gt, n_classes, k=16)
    print(f'Cache-adapter:    {ca_time*1000:.4f} ms/query (cache size={cache_size} examples, '
          f'K=16/class)')

    knn_time, train_size = time_knn(feats, gt, n_classes)
    print(f'kNN probe:        {knn_time*1000:.4f} ms/query (reference set={train_size} examples, '
          f'~80% of cohort)')

    clust_time, n_total = time_clustering(feats, gt, n_classes)
    print(f'Clustering:       {clust_time*1000:.4f} ms for ONE new point '
          f'(requires refitting k-means over all {n_total} points -- no incremental '
          f'single-query inference exists for this method)')

    print()
    print('=== Relative cost (vs zero-shot) ===')
    print(f'Cache-adapter: {ca_time/zs_time:.1f}x zero-shot')
    print(f'kNN:           {knn_time/zs_time:.1f}x zero-shot')
    print(f'Clustering:    {clust_time/zs_time:.0f}x zero-shot (one-shot batch cost, not per-query)')


if __name__ == '__main__':
    main()
