"""
Experiment 1 — Baseline: local patch-comparison (PatchCore-style) under a
ONE-SHOT constraint: the "memory bank" is built from a single reference
image per category instead of the usual ~350 training images.

Goal: show this catches surface/structural damage reasonably but is weak
on logical anomalies (missing-part regions still look locally normal).

Run:  python exp1_patchcore_oneshot.py
Needs: torch, torchvision, scikit-learn, pillow
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from utils import CATEGORIES, pick_one_shot_reference, build_eval_set, report_auroc

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- feature extractor ----------------
_model = wide_resnet50_2(weights="IMAGENET1K_V2").to(device).eval()
_feats = {}

def _hook(name):
    def fn(module, inp, out):
        _feats[name] = out
    return fn

_model.layer2.register_forward_hook(_hook("l2"))
_model.layer3.register_forward_hook(_hook("l3"))

_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def patch_features(img_path):
    x = _tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    _model(x)
    f2 = F.avg_pool2d(_feats["l2"], 3, 1, 1)
    f3 = F.interpolate(_feats["l3"], size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f3 = F.avg_pool2d(f3, 3, 1, 1)
    feat = torch.cat([f2, f3], dim=1)               # (1, C, H, W)
    _, C, H, W = feat.shape
    flat = feat.squeeze(0).permute(1, 2, 0).reshape(H * W, C).cpu().numpy()
    return flat, (H, W)


def run_category(category):
    ref_path = pick_one_shot_reference(category)
    print(f"\n=== {category} | reference image: {ref_path} ===")

    ref_bank, _ = patch_features(ref_path)
    nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ref_bank)

    images, labels, kind = build_eval_set(category)
    print(f"  {len(images)} test images to score...")
    scores = []
    for i, img_path in enumerate(images, 1):
        q, _ = patch_features(img_path)
        dist, _ = nn_index.kneighbors(q)
        scores.append(float(dist.max()))            # image score = worst-matching patch
        print(f"  [{i}/{len(images)}] {kind[i-1]:>10s}  score={scores[-1]:.4f}  {img_path}", flush=True)

    return report_auroc(scores, labels, kind, name=f"Exp1-PatchCore-1shot/{category}")


if __name__ == "__main__":
    print(f"Device: {device}")
    results = {}
    for ci, cat in enumerate(CATEGORIES, 1):
        print(f"\n[{ci}/{len(CATEGORIES)}] category: {cat}")
        try:
            results[cat] = run_category(cat)
        except AssertionError as e:
            print(f"Skipping {cat}: {e}")

    if results:
        import numpy as np
        for key in ["overall", "logical", "structural"]:
            vals = [r[key] for r in results.values() if not np.isnan(r[key])]
            if vals:
                print(f"\nAVERAGE {key}: {np.mean(vals):.3f}")