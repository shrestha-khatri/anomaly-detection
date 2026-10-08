"""
Physically enforces the one-shot constraint on a saved MVTec LOCO AD folder:
  - keeps exactly ONE image per category in train/good (the one utils.py's
    pick_one_shot_reference() would select — same seed, same logic, so the
    reference used by your experiments doesn't change)
  - moves every other train/good image out to a sibling
    "<DATASET_ROOT>_quarantine/<category>/train/good_unused/" folder
  - moves a validation/ split out the same way, if one exists (the original
    LOCO benchmark uses it for threshold calibration on real images — using
    it here would violate "no anomaly-adjacent data used to derive the
    threshold", since it's a form of held-out ground truth about normality
    beyond your single reference)

Nothing is deleted — everything is moved to the quarantine folder, so you
can restore it later if needed. Run with --dry-run first to see exactly
what would move before it actually moves anything.

USAGE
-----
    python enforce_one_shot.py --dry-run     # see what would happen
    python enforce_one_shot.py               # actually move the files

Uses the same DATASET_ROOT and pick_one_shot_reference() as utils.py, so
run this from the same directory (or make sure utils.py is importable).
"""

import argparse
import shutil
from pathlib import Path

from utils import DATASET_ROOT, CATEGORIES, category_paths, pick_one_shot_reference


def quarantine_dir(dataset_root):
    return Path(str(dataset_root).rstrip("/\\") + "_quarantine")


def enforce_category(category, dataset_root, dry_run):
    paths = category_paths(category, dataset_root)
    train_good = paths["train_good"]
    if not train_good:
        print(f"  [{category}] no train/good images found — skipping")
        return

    reference = pick_one_shot_reference(category, dataset_root)
    reference_name = Path(reference).name
    quarantine_base = quarantine_dir(dataset_root) / category / "train" / "good_unused"

    to_move = [p for p in train_good if Path(p).name != reference_name]
    print(f"  [{category}] reference kept: {reference}")
    print(f"  [{category}] {len(to_move)} other train/good image(s) will be "
          f"{'moved (dry run — nothing happens yet)' if dry_run else 'moved'} to:")
    print(f"    {quarantine_base}")

    if not dry_run:
        quarantine_base.mkdir(parents=True, exist_ok=True)
        for src in to_move:
            dst = quarantine_base / Path(src).name
            shutil.move(src, dst)

    # validation split, if present
    val_dir = Path(dataset_root) / category / "validation"
    if val_dir.exists():
        val_quarantine = quarantine_dir(dataset_root) / category / "validation"
        print(f"  [{category}] validation/ split found — will be "
              f"{'moved (dry run)' if dry_run else 'moved'} to {val_quarantine}")
        if not dry_run:
            val_quarantine.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(val_dir), str(val_quarantine))
    else:
        print(f"  [{category}] no validation/ split found — nothing to do there")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would be moved without moving anything.")
    parser.add_argument("--dataset-root", default=DATASET_ROOT,
                         help="Override utils.DATASET_ROOT if needed.")
    args = parser.parse_args()

    print(f"Dataset root: {args.dataset_root}")
    print(f"Quarantine folder: {quarantine_dir(args.dataset_root)}")
    print(f"Mode: {'DRY RUN (nothing will move)' if args.dry_run else 'LIVE (files will move)'}\n")

    for category in CATEGORIES:
        enforce_category(category, args.dataset_root, args.dry_run)
        print()

    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to actually quarantine the files.")
    else:
        print("Done. Your dataset folder now contains only the single locked "
              "reference image per category in train/good. Nothing was deleted — "
              "everything moved to the quarantine folder above if you need it back.")


if __name__ == "__main__":
    main()