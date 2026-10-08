"""
Shared utilities for the three one-shot logical-anomaly experiments.

Expected MVTec LOCO AD layout (official layout, adjust DATASET_ROOT below):

DATASET_ROOT/
  <category>/
    train/good/*.png
    test/good/*.png
    test/logical_anomalies/*.png
    test/structural_anomalies/*.png
    ground_truth/logical_anomalies/<image_stem>/000.png
    ground_truth/structural_anomalies/<image_stem>/000.png
"""

import os
import glob
import random
import numpy as np
from sklearn.metrics import roc_auc_score

# ---- EDIT THIS to point at your saved dataset folder ----
DATASET_ROOT = "D:\Download\ie\data"
CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins",
              "screw_bag", "splicing_connectors"]


def category_paths(category, root=DATASET_ROOT):
    base = os.path.join(root, category)
    return {
        "train_good": sorted(glob.glob(os.path.join(base, "train", "good", "*.png"))),
        "test_good": sorted(glob.glob(os.path.join(base, "test", "good", "*.png"))),
        "test_logical": sorted(glob.glob(os.path.join(base, "test", "logical_anomalies", "*.png"))),
        "test_structural": sorted(glob.glob(os.path.join(base, "test", "structural_anomalies", "*.png"))),
        "gt_logical_dir": os.path.join(base, "ground_truth", "logical_anomalies"),
        "gt_structural_dir": os.path.join(base, "ground_truth", "structural_anomalies"),
    }


def pick_one_shot_reference(category, root=DATASET_ROOT, seed=0):
    """The ONE normal image per category used as the sole reference.
    Fixed seed so every experiment uses the identical reference image."""
    paths = category_paths(category, root)
    assert len(paths["train_good"]) > 0, f"No train/good images found for {category} under {root}"
    rng = random.Random(seed)
    return rng.choice(paths["train_good"])


def gt_mask_path(gt_dir, image_path):
    """MVTec LOCO stores one ground-truth folder per anomalous image,
    containing one or more binary masks (000.png, 001.png, ...)."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    folder = os.path.join(gt_dir, stem)
    masks = sorted(glob.glob(os.path.join(folder, "*.png")))
    return masks  # list, may be empty if no gt for that split


def build_eval_set(category, root=DATASET_ROOT):
    """Returns (image_paths, image_labels) for image-level AUROC:
    label 0 = good, 1 = anomalous (logical + structural pooled),
    plus separate index lists to slice logical-only / structural-only later."""
    paths = category_paths(category, root)
    images, labels, kind = [], [], []

    for p in paths["test_good"]:
        images.append(p); labels.append(0); kind.append("good")
    for p in paths["test_logical"]:
        images.append(p); labels.append(1); kind.append("logical")
    for p in paths["test_structural"]:
        images.append(p); labels.append(1); kind.append("structural")

    return images, np.array(labels), np.array(kind)


def report_auroc(scores, labels, kind, name=""):
    """Prints overall + logical-only + structural-only image AUROC."""
    scores = np.array(scores)
    overall = roc_auc_score(labels, scores)

    log_mask = (kind == "good") | (kind == "logical")
    struct_mask = (kind == "good") | (kind == "structural")

    log_auc = roc_auc_score(labels[log_mask], scores[log_mask]) if log_mask.sum() else float("nan")
    struct_auc = roc_auc_score(labels[struct_mask], scores[struct_mask]) if struct_mask.sum() else float("nan")

    print(f"[{name}] Image AUROC  overall={overall:.3f}  logical={log_auc:.3f}  structural={struct_auc:.3f}")
    return {"overall": overall, "logical": log_auc, "structural": struct_auc}