"""
Experiment 2 — Global-arrangement model: segment components in the single
reference image and in each test image, then compare COUNTS and RELATIVE
POSITIONS instead of raw local patches. This is the seed of a mechanism
that "reasons about the overall arrangement" rather than local texture.

Two building blocks are given:
  1. background-color-distance segmentation + connected components —
     fast, no training; more stable than a global Otsu threshold on these
     tray/table backgrounds (Otsu flips foreground/background depending on
     lighting and clutter, which produces wildly inconsistent object counts
     even across normal images — the background-distance version fixes that
     by locking the background color estimate to the one reference image)
  2. a simple graph-based comparison: each image -> set of (x, y, area)
     nodes; compare via count difference + Hungarian-matched displacement

`run_category` writes `debug_mask_<category>.png` (mask | original side by
side) for the reference image on every run — always look at this first.
Tune segmentation per category via `CATEGORY_SEG_PARAMS` at the top of the
file: `dist_thresh` (lower = more sensitive to subtle color differences from
background), `close_kernel`/`close_iters` (how aggressively nearby blobs get
fused — too high and genuinely separate objects merge into one giant blob,
too low and one object fragments into several pieces), `min_area_frac`
(raise to drop small noise blobs).

Swap step 1 for a proper instance segmenter (e.g. SAM) per category if
background-distance thresholding is still too crude for a given product.

Run:  python exp2_component_arrangement.py
Needs: opencv-python, scipy, numpy
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import CATEGORIES, pick_one_shot_reference, build_eval_set, report_auroc


def estimate_background_color(img, border_frac=0.03):
    """Sample a thin border strip around the image and take the median color.
    Works for tray/table-style backgrounds (pushpins, screw_bag, splicing_connectors)
    where the background is roughly uniform and objects sit on top of it."""
    h, w = img.shape[:2]
    bh, bw = max(2, int(h * border_frac)), max(2, int(w * border_frac))
    strips = np.concatenate([
        img[:bh, :].reshape(-1, 3),
        img[-bh:, :].reshape(-1, 3),
        img[:, :bw].reshape(-1, 3),
        img[:, -bw:].reshape(-1, 3),
    ], axis=0)
    return np.median(strips, axis=0)


def get_components(img_path, min_area_frac=0.0006, bg_color=None, dist_thresh=30,
                    close_kernel=3, close_iters=1, open_iters=1,
                    split_touching=False, min_peak_distance=15):
    """Segment foreground components by distance from a background color.

    split_touching=True runs watershed segmentation (distance-transform +
    peak markers) on the foreground mask, which splits objects that are
    physically touching or overlapping into separate components. Plain
    connected-components CANNOT do this — touching objects form a single
    connected region no matter how threshold/morphology are tuned, which is
    why n_comps was collapsing to 1 for screw_bag/splicing_connectors even
    after fixing the closing kernel. Use this for categories where objects
    pile up or touch; leave off where it would over-split well-separated
    objects. min_peak_distance is roughly the minimum expected gap in pixels
    between two distinct object centers — tune this per category.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img_f = img.astype(np.float32)

    if bg_color is None:
        bg_color = estimate_background_color(img)

    dist = np.linalg.norm(img_f - bg_color.reshape(1, 1, 3), axis=-1)
    mask = (dist > dist_thresh).astype(np.uint8) * 255

    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    if open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    min_area = min_area_frac * img.shape[0] * img.shape[1]

    if not split_touching:
        n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
        return [(cent[i], stats[i, cv2.CC_STAT_AREA])
                for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > min_area]

    # --- watershed: split touching/overlapping objects within the mask ---
    binary = mask > 0
    if not binary.any():
        return []
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=min_peak_distance, labels=binary)
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    ws_labels = watershed(-distance, markers, mask=binary)

    comps = []
    for label_id in range(1, ws_labels.max() + 1):
        ys, xs = np.where(ws_labels == label_id)
        area = len(xs)
        if area > min_area:
            comps.append(((float(xs.mean()), float(ys.mean())), area))
    return comps


# Per-category overrides — start here when tuning. dist_thresh: lower = more
# sensitive to subtle color differences from background (catches more, but
# noisier). close_kernel/close_iters: raise only if a single object is being
# split into several pieces; lower if separate objects are fusing into one
# (but note: if objects are actually TOUCHING in the image, no amount of
# closing/opening will separate them — use split_touching instead).
# min_peak_distance (watershed only): roughly the smallest gap you'd expect
# between two real object centers — too small over-splits single objects,
# too large under-splits touching clusters.
CATEGORY_SEG_PARAMS = {
    "pushpins":            dict(dist_thresh=25, close_kernel=3, close_iters=1, min_area_frac=0.0004,
                                 split_touching=True, min_peak_distance=20),
    "screw_bag":           dict(dist_thresh=20, close_kernel=3, close_iters=0, min_area_frac=0.0002,
                                 split_touching=True, min_peak_distance=12),
    "splicing_connectors": dict(dist_thresh=25, close_kernel=3, close_iters=1, min_area_frac=0.0004,
                                 split_touching=True, min_peak_distance=25),
    "breakfast_box":       dict(dist_thresh=30, close_kernel=3, close_iters=1, min_area_frac=0.0006),
    "juice_bottle":        dict(dist_thresh=30, close_kernel=3, close_iters=1, min_area_frac=0.0006),
}


def arrangement_score(ref_comps, test_comps, img_diag=400.0):
    """Returns an anomaly score combining:
       - count mismatch (missing/extra components)
       - matched-centroid displacement (wrong position / arrangement)
    Normalized so it behaves like a distance (higher = more anomalous)."""
    n_ref, n_test = len(ref_comps), len(test_comps)
    count_penalty = abs(n_ref - n_test) / max(n_ref, 1)

    if n_ref == 0 or n_test == 0:
        return 1.0 + count_penalty

    ref_xy = np.array([c[0] for c in ref_comps])
    test_xy = np.array([c[0] for c in test_comps])

    # cost matrix = pairwise euclidean distance, normalized by image diagonal
    cost = np.linalg.norm(ref_xy[:, None, :] - test_xy[None, :, :], axis=-1) / img_diag
    r_idx, c_idx = linear_sum_assignment(cost)
    displacement_penalty = cost[r_idx, c_idx].mean()

    return count_penalty + displacement_penalty


def _save_debug_mask(img_path, bg_color, out_path, seg_params):
    """Writes the original image with a marker + index at every detected
    component centroid, so you can directly count/compare against the real
    object count instead of squinting at a raw binary mask."""
    img = cv2.imread(img_path)
    comps = get_components(img_path, bg_color=bg_color, **seg_params)
    vis = img.copy()
    for i, (centroid, area) in enumerate(comps):
        x, y = int(centroid[0]), int(centroid[1])
        cv2.drawMarker(vis, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                        markerSize=14, thickness=2)
        cv2.putText(vis, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"n_comps={len(comps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def run_category(category, save_debug_mask=True):
    ref_path = pick_one_shot_reference(category)
    print(f"\n=== {category} | reference image: {ref_path} ===")

    seg_params = CATEGORY_SEG_PARAMS.get(category, {})

    ref_img = cv2.imread(ref_path)
    diag = float(np.hypot(*ref_img.shape[:2]))
    bg_color = estimate_background_color(ref_img)
    ref_comps = get_components(ref_path, bg_color=bg_color, **seg_params)
    print(f"  estimated background color (BGR): {bg_color}  |  seg_params={seg_params}")

    if save_debug_mask:
        _save_debug_mask(ref_path, bg_color, f"debug_mask_{category}.png", seg_params)
        print(f"  wrote debug_mask_{category}.png — open it to sanity-check the segmentation "
              f"before trusting the scores; if it looks like noise or one giant blob, tune "
              f"CATEGORY_SEG_PARAMS['{category}'] at the top of this file")

    images, labels, kind = build_eval_set(category)
    print(f"  {len(images)} test images to score... (ref has {len(ref_comps)} components)")
    scores = []
    for i, img_path in enumerate(images, 1):
        test_comps = get_components(img_path, bg_color=bg_color, **seg_params)
        scores.append(arrangement_score(ref_comps, test_comps, img_diag=diag))
        print(f"  [{i}/{len(images)}] {kind[i-1]:>10s}  score={scores[-1]:.4f}  "
              f"n_comps={len(test_comps)}  {img_path}", flush=True)

    return report_auroc(scores, labels, kind, name=f"Exp2-Arrangement/{category}")


if __name__ == "__main__":
    results = {}
    for ci, cat in enumerate(CATEGORIES, 1):
        print(f"\n[{ci}/{len(CATEGORIES)}] category: {cat}")
        try:
            results[cat] = run_category(cat)
        except AssertionError as e:
            print(f"Skipping {cat}: {e}")

    if results:
        for key in ["overall", "logical", "structural"]:
            vals = [r[key] for r in results.values() if not np.isnan(r[key])]
            if vals:
                print(f"\nAVERAGE {key}: {np.mean(vals):.3f}")