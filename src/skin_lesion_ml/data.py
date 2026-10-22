"""Explicit feature-table contracts; metadata never enters the estimator."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["mel", "nv", "vasc"]
METADATA = {"image_id", "aug_tag", "dx", "filename", "patient_id", "lesion_id"}


def load_table(path, group_column="image_id", label_column="dx", feature_names=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names are not allowed.")
    frame = pd.read_csv(path, dtype={group_column: str})
    if frame.empty:
        raise ValueError("The feature table is empty.")
    if feature_names is not None:  # inference: labels and groups are optional
        missing = set(feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        names = feature_names
    else:
        for col in (group_column, label_column):
            if col not in frame or frame[col].isna().any():
                raise ValueError(f"Required column is missing or contains nulls: {col}")
        labels = set(frame[label_column])
        if labels != set(CLASSES):
            raise ValueError(f"Expected all three labels {CLASSES}; got {sorted(map(str, labels))}")
        if (frame.groupby(group_column)[label_column].nunique() != 1).any():
            raise ValueError("Each group must have exactly one diagnosis.")
        excluded = METADATA | {group_column, label_column}
        names = [col for col in frame if col not in excluded]
    if not names:
        raise ValueError("No feature columns found.")
    X = frame[names].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite feature values are not allowed; missing values may be NaN.")
    return frame, X


def make_demo(path, seed=42):
    """Synthetic groups with correlated views, never a medical benchmark."""
    rng = np.random.default_rng(seed)
    rows = []
    names = ([f"int_Lab_a_stat{i}" for i in range(6)]
             + [f"glcm_contrast_d1_a{i}" for i in range(6)]
             + [f"asym_stat{i}" for i in range(6)])
    for class_id, label in enumerate(CLASSES):
        for group in range(15):
            latent = rng.normal(size=len(names))
            latent[:6] += (class_id == 2) * 2.2
            latent[6:12] += (class_id == 0) * 1.3
            for view in range(3):
                values = latent + rng.normal(0, 0.08, len(names))
                rows.append(dict(image_id=f"demo_{label}_{group:03}", dx=label,
                                 aug_tag=f"view{view}", **dict(zip(names, values))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
