"""
Experiment 3 — Foundation-model one-shot baseline, WinCLIP-style.

Two scoring signals, combined:
  (a) visual: cosine distance between test-image CLIP embedding and the
      SINGLE reference image's CLIP embedding (image-image, true one-shot)
  (b) textual: CLIP zero-shot similarity to a small prompt ensemble of
      "normal" vs "damaged/incomplete/misassembled" style phrases
      (captures the WinCLIP idea of language-guided anomaly scoring)

This needs open_clip (pip install open_clip_torch). If you'd rather use
the full windowed/patch-level WinCLIP with segmentation maps, swap this
for anomalib's WinClipModel (pip install anomalib) — same reference-image
input, but it also gives a pixel-level anomaly map, not just a score.

Run:  python exp3_clip_oneshot.py
"""

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

from utils import CATEGORIES, pick_one_shot_reference, build_eval_set, report_auroc

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-16")
model = model.to(device).eval()

NORMAL_PROMPTS = [
    "a photo of a normal {obj} with all parts correctly assembled",
    "a photo of a flawless {obj}",
    "a photo of a {obj} in perfect condition",
]
ANOMALY_PROMPTS = [
    "a photo of a {obj} with a missing part",
    "a photo of a {obj} with a wrong number of components",
    "a photo of a {obj} with parts in the wrong place",
    "a photo of a damaged {obj}",
]


@torch.no_grad()
def clip_image_embed(img_path):
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    return F.normalize(feat, dim=-1)


@torch.no_grad()
def clip_text_embed(prompts, obj_name):
    texts = tokenizer([p.format(obj=obj_name.replace("_", " ")) for p in prompts]).to(device)
    feat = model.encode_text(texts)
    feat = F.normalize(feat, dim=-1)
    return feat.mean(dim=0, keepdim=True)  # prompt-ensemble average


def run_category(category):
    ref_path = pick_one_shot_reference(category)
    print(f"\n=== {category} | reference image: {ref_path} ===")

    ref_embed = clip_image_embed(ref_path)
    normal_text = clip_text_embed(NORMAL_PROMPTS, category)
    anomaly_text = clip_text_embed(ANOMALY_PROMPTS, category)

    images, labels, kind = build_eval_set(category)
    print(f"  {len(images)} test images to score...")
    scores = []
    for i, img_path in enumerate(images, 1):
        q = clip_image_embed(img_path)

        # (a) visual: distance from the one reference image (higher = more anomalous)
        visual_dist = 1.0 - (q @ ref_embed.T).item()

        # (b) textual: how much closer to "anomaly" prompts than "normal" prompts
        sim_normal = (q @ normal_text.T).item()
        sim_anom = (q @ anomaly_text.T).item()
        text_score = sim_anom - sim_normal

        scores.append(visual_dist + text_score)
        print(f"  [{i}/{len(images)}] {kind[i-1]:>10s}  score={scores[-1]:.4f}  {img_path}", flush=True)

    return report_auroc(scores, labels, kind, name=f"Exp3-CLIP-1shot/{category}")


if __name__ == "__main__":
    import numpy as np
    print(f"Device: {device}")
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