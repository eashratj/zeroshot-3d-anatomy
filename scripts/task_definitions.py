"""
Single source of truth for T1-T6's class definitions: which folders map
to which label, which point-cloud root each folder lives under, and the
natural-language class name used for zero-shot text prompts. Built from
the verified category list in cohort277_meta/tasks_categories.txt (every
name checked against the live MedShapeNet index before being added here
-- see build notes in that dir).

Two point-cloud roots are in play:
  COHORT_ROOT  -- the 277-subject cohort (T1-T5, and T6's healthy side),
                  written by build_cohort277.py.
  LEGACY_ROOT  -- the original paper's already-preprocessed data (T6's
                  tumor side: tumoredKidney/tumoredbrain, which have
                  near-zero subject-ID overlap with the 277-cohort --
                  confirmed cross-source, KiTS21/BraTS -- so they were
                  never going to be re-downloaded via build_cohort277).

T2 laterality is deliberately NOT one 15-class task -- per the
per-structure design decision, it's 15 independent binary (left vs
right) tasks, one per structure type, each scored and reported
separately rather than pooled into one number.

Each task exposes:
  folder_label_root: {folder_name: (label_idx, root_path)}
  class_names: [name per label_idx, in index order] -- for zero-shot
    text-prompt templates.
"""

from pathlib import Path

COHORT_ROOT = Path('/msc/home/q45emuni/foundation_models/shapemi/pointclouds')
LEGACY_ROOT = Path('/msc/home/q45emuni/miccai_2026/shapemi/pointclouds')


def _cohort(*folders):
    return {f: COHORT_ROOT for f in folders}


# ---------------------------------------------------------------- T1 ----
T1_ORGAN_ID = dict(
    folder_label_root={
        'liver': (0, COHORT_ROOT), 'heart': (1, COHORT_ROOT),
        'trachea': (2, COHORT_ROOT), 'esophagus': (3, COHORT_ROOT),
        'colon': (4, COHORT_ROOT), 'duodenum': (5, COHORT_ROOT),
        'bladder': (6, COHORT_ROOT), 'gallbladder': (7, COHORT_ROOT),
        # kidney pools left+right into one "kidney" class for organ-ID
        # (laterality is T2's question, not T1's)
        'kidneyleft': (8, COHORT_ROOT), 'kidneyright': (8, COHORT_ROOT),
    },
    class_names=['liver', 'heart', 'trachea', 'esophagus', 'colon',
                 'duodenum', 'bladder', 'gallbladder', 'kidney'],
)

# ---------------------------------------------------------------- T2 ----
# 15 independent binary left-vs-right tasks. Natural-language name is the
# same for both sides (laterality isn't part of the class name -- that's
# exactly what the model has to infer from shape/position alone).
T2_STRUCTURES = {
    'adrenalgland': 'adrenal gland', 'autochthon': 'autochthon muscle',
    'clavicle': 'clavicle', 'femur': 'femur',
    'gluteusmaximus': 'gluteus maximus', 'gluteusmedius': 'gluteus medius',
    'gluteusminimus': 'gluteus minimus', 'hip': 'hip bone',
    'humerus': 'humerus', 'iliacartery': 'iliac artery',
    'iliacvena': 'iliac vein', 'iliopsoas': 'iliopsoas muscle',
    'kidney': 'kidney', 'thyroid': 'thyroid', 'eyeball': 'eyeball',
}


def t2_laterality_task(structure_key: str):
    """structure_key: one of T2_STRUCTURES' keys. Returns a task dict for
    that single structure's left-vs-right binary task."""
    name = T2_STRUCTURES[structure_key]
    left_folder = f'{structure_key}left'
    right_folder = f'{structure_key}right'
    return dict(
        folder_label_root={left_folder: (0, COHORT_ROOT),
                            right_folder: (1, COHORT_ROOT)},
        class_names=[f'left {name}', f'right {name}'],
    )


T2_ALL_LATERALITY_TASKS = {k: t2_laterality_task(k) for k in T2_STRUCTURES}

# ---------------------------------------------------------------- T3 ----
T3_LUNG_LOBES = dict(
    folder_label_root={
        'lungupperloberight': (0, COHORT_ROOT),
        'lungupperlobeleft': (1, COHORT_ROOT),
        'lungmiddleloberight': (2, COHORT_ROOT),
        'lunglowerloberight': (3, COHORT_ROOT),
        'lunglowerlobeleft': (4, COHORT_ROOT),
    },
    class_names=['right upper lung lobe', 'left upper lung lobe',
                 'right middle lung lobe', 'right lower lung lobe',
                 'left lower lung lobe'],
)

# ---------------------------------------------------------------- T4 ----
T4_CARDIAC_CHAMBER = dict(
    folder_label_root={
        'heartatriumleft': (0, COHORT_ROOT),
        'heartatriumright': (1, COHORT_ROOT),
        'heartventricleleft': (2, COHORT_ROOT),
        'heartventricleright': (3, COHORT_ROOT),
        'heartmyocardium': (4, COHORT_ROOT),
    },
    class_names=['left atrium', 'right atrium', 'left ventricle',
                 'right ventricle', 'myocardium'],
)

# ---------------------------------------------------------------- T5 ----
_VERT_ORDER = ([f'C{i}' for i in range(1, 8)] +
               [f'T{i}' for i in range(1, 13)] +
               [f'L{i}' for i in range(1, 6)])
T5_VERTEBRAL_LEVEL = dict(
    folder_label_root={f'vertebrae{v}': (i, COHORT_ROOT)
                        for i, v in enumerate(_VERT_ORDER)},
    class_names=[f'{v} vertebra' for v in _VERT_ORDER],
)
# 3-view reporting per spec Sec. 7.4: exact (24-way, above), 3-way region,
# and adjacent-level confusion (computed downstream from the 24-way
# confusion matrix, not a separate task definition).
T5_REGION_OF = {v: ('cervical' if v.startswith('C') else
                     'thoracic' if v.startswith('T') else 'lumbar')
                 for v in _VERT_ORDER}

# ---------------------------------------------------------------- T6 ----
# Healthy side re-sourced to the 277-cohort (decision, 2026-09-14).
# Tumor side stays on LEGACY_ROOT -- unavoidably cross-source
# (KiTS21/BraTS), confirmed near-zero subject-ID overlap with the cohort.
T6_KIDNEY_PATHOLOGY = dict(
    folder_label_root={
        'kidneyleft': (0, COHORT_ROOT), 'kidneyright': (0, COHORT_ROOT),
        'tumoredKidney': (1, LEGACY_ROOT),
    },
    class_names=['normal kidney', 'kidney with a tumor'],
)
T6_BRAIN_PATHOLOGY = dict(
    folder_label_root={
        'brain': (0, COHORT_ROOT),
        'tumoredbrain': (1, LEGACY_ROOT),
    },
    class_names=['normal brain', 'brain with a tumor'],
)

ALL_TASKS = {
    'T1_organ_id': T1_ORGAN_ID,
    **{f'T2_{k}': v for k, v in T2_ALL_LATERALITY_TASKS.items()},
    'T3_lung_lobes': T3_LUNG_LOBES,
    'T4_cardiac': T4_CARDIAC_CHAMBER,
    'T5_vertebrae': T5_VERTEBRAL_LEVEL,
    'T6_kidney_pathology': T6_KIDNEY_PATHOLOGY,
    'T6_brain_pathology': T6_BRAIN_PATHOLOGY,
}


def collect_samples_multiroot(folder_label_root: dict):
    """Like each per-model lib's own collect_samples, but supports a
    different POINTCLOUD_ROOT per folder (needed for T6's mixed-source
    healthy/tumor data). Returns list of (npy_path, label_idx)."""
    samples = []
    for folder, (label, root) in folder_label_root.items():
        folder_dir = root / folder
        if not folder_dir.exists():
            print(f'  WARNING: {folder_dir} does not exist yet, skipping')
            continue
        for f in sorted(folder_dir.glob('*.npy')):
            samples.append((f, label))
    return samples


if __name__ == '__main__':
    # sanity-print: every task's class count and (once data has landed)
    # sample count, so a broken/missing folder is obvious immediately
    # rather than surfacing as a cryptic downstream shape mismatch.
    for name, task in ALL_TASKS.items():
        samples = collect_samples_multiroot(task['folder_label_root'])
        n_classes = len(task['class_names'])
        print(f'{name}: {n_classes} classes, {len(samples)} samples total')
