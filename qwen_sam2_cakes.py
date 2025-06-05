"""qwen_sam2_cakes.py
====================
Fixed import path: **segment_anything** instead of sam2.modeling**, so Vit‑H
weights work out‑of‑the‑box.**

Run example:
```bash
python qwen_sam2_cakes.py \
  --gpus 0,1,2,3 \
  --image ./assets/spatial_understanding/cakes.png \
  --qwen_dir ./qwen_vl \
  --sam_ckpt ./sam_ckpt/sam_vit_h_4b8939.pth \
  --out overlay
```
Make sure you installed the original SAM package:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```
"""

# ─────────────── early GPU parsing (unchanged) ────────────────
import os, sys
if "--gpus" in sys.argv:
    i = sys.argv.index("--gpus")
    if i + 1 >= len(sys.argv):
        raise RuntimeError("--gpus requires a value, e.g. --gpus 0,1,2")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
    sys.argv.pop(i + 1)
    sys.argv.pop(i)

# ───────────────────────── imports ─────────────────────────────
import argparse, json, re
from pathlib import Path
from typing import List, Dict

import cv2, numpy as np, torch, matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
# **Fixed here**
from segment_anything import SamPredictor, sam_model_registry

# ─────────────────────── helper utils ─────────────────────────

def load_qwen(model_dir: str):
    """Load Qwen with automatic GPU/CPU dispatch.
    * If multiple GPUs are visible → weights均匀切片 (device_map="auto").
    * 如果只有一张卡不足 → 自动把多余权重 offload 到 CPU。
    """
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",          # <-- 核心改动：让 HF 自动在多卡/CPU 间分配
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return proc, model


def ask_for_points(proc, model, pil_img: Image.Image) -> List[Dict[str, int]]:
    prompt = (
        "Identify the approximate center point of **each** cake in the picture. "
        "Return a JSON list where each element is {'x': int, 'y': int}."
    )
    inputs = proc(text=prompt, images=pil_img, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128)
    txt = proc.tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r"\[[\s\S]*\]", txt)
    if not m:
        raise ValueError(f"Qwen did not return JSON list, got: {txt}")
    pts = json.loads(m.group(0))
    return [{"x": int(p["x"]), "y": int(p["y"])} for p in pts]


# ───────────────────── SAM (v1) helpers ──────────────────────

def load_sam(ckpt: str, model_type: str = "vit_h"):
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    return SamPredictor(sam)


def segment_masks(pred: SamPredictor, img_bgr: np.ndarray, pts):
    pred.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    masks, scores = [], []
    for p in pts:
        m, s, _ = pred.predict(
            point_coords=np.array([[p["x"], p["y"]]]),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=False,
        )
        masks.append(m[0])
        scores.append(float(s[0]))
    return masks, scores


# ─────────────────── visualisation utils ─────────────────────

def draw_points(img: np.ndarray, pts, radius=6):
    vis = img.copy()
    for p in pts:
        cv2.circle(vis, (p["x"], p["y"]), radius, (0, 0, 255), -1)
    return vis


def overlay_masks(base: np.ndarray, mask_list, alpha=0.4):
    out = base.copy()
    cmap = plt.cm.get_cmap("Set3", len(mask_list))
    for idx, m in enumerate(mask_list):
        col = tuple(int(c * 255) for c in cmap(idx)[:3])[::-1]  # BGR
        layer = np.zeros_like(base)
        layer[m] = col
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    return out

# ─────────────────────────── main ────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--qwen_dir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--out", default="overlay")
    args = ap.parse_args()

    out_png = Path(f"{Path(args.out).stem}.png").resolve()
    dots_png = Path("cakes_with_points.png").resolve()

    img_path = Path(args.image).expanduser().resolve()
    pil = Image.open(img_path).convert("RGB")
    bgr = cv2.imread(str(img_path))

    proc, qwen = load_qwen(args.qwen_dir)
    pts = ask_for_points(proc, qwen, pil)
    print("[Qwen]", len(pts), "cakes →", pts)

    Image.fromarray(draw_points(np.array(pil), pts)).save(dots_png)
    print("Dots saved →", dots_png.name)

    sam_pred = load_sam(args.sam_ckpt)
    masks, ious = segment_masks(sam_pred, bgr, pts)
    print("[SAM] IoU:", [f"{s:.3f}" for s in ious])

    cv2.imwrite(str(out_png), overlay_masks(bgr, masks))
    print("Overlay saved →", out_png.name)


if __name__ == "__main__":
    main()
