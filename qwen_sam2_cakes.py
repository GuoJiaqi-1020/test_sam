"""qwen_sam2_cakes.py
====================
End‑to‑end demo that
1. loads an input image (default: ./assets/spatial_understanding/cakes.png),
2. uses **Qwen‑2.5‑VL** to output (x, y) center‑points for *all* cakes,
3. draws red dots on the original image and saves **cakes_with_points.png** in the *current working directory*,
4. feeds each point to **SAM** (Vit‑H by default) to get instance masks, and
5. overlays every mask on the original image with 40 % transparency, saving **overlay.png** (or your custom name, always PNG) in the *current working directory*.

Usage (after env + weights ready):

```bash
python qwen_sam2_cakes.py \
  --image ./assets/spatial_understanding/cakes.png \
  --qwen_dir ./qwen_vl \
  --sam_ckpt ./sam_ckpt/sam_vit_h_4b8939.pth \
  --out custom_overlay   # optional, .png suffix auto‑added
```
"""

import argparse, json, re
from pathlib import Path
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.modeling import SamPredictor
from transformers import AutoModelForVision2Seq, AutoProcessor

# ───────────────────────── Qwen helpers ────────────────────────── #

def load_qwen(model_dir: str, device: str = "cuda"):
    """Return (processor, model) on chosen device; FP16 for memory‑safe GPU."""
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    return proc, model


def ask_for_points(proc, model, pil_img: Image.Image) -> List[Dict[str, int]]:
    """Let Qwen output JSON list of center points for every cake."""
    prompt = (
        "Identify the approximate center point of **each** cake in the picture. "
        "Return a JSON list where each element is {{'x': int, 'y': int}} in *original* pixels."
    )
    inputs = proc(text=prompt, images=pil_img, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128)
    txt = proc.tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r"\[[\s\S]*\]", txt)
    if not m:
        raise ValueError(f"Qwen did not return JSON list, got: {txt}")
    pts = json.loads(m.group(0))
    if not isinstance(pts, list):
        raise ValueError("Parsed JSON is not a list")
    return [{"x": int(p["x"]), "y": int(p["y"])} for p in pts]

# ───────────────────────── SAM helpers ────────────────────────── #

def load_sam(ckpt: str, model_type: str = "vit_h"):
    return SamPredictor(checkpoint=ckpt, model_type=model_type)


def segment_masks(predictor: SamPredictor, img_bgr: np.ndarray, pts):
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    masks, scores = [], []
    for p in pts:
        coord = np.array([[p["x"], p["y"]]])
        m, s, _ = predictor.predict(coord, np.array([1], dtype=np.int32), multimask_output=False)
        masks.append(m[0])
        scores.append(float(s[0]))
    return masks, scores

# ───────────────────────── visual helpers ──────────────────────── #

def draw_points(img: np.ndarray, pts, radius: int = 6):
    vis = img.copy()
    for p in pts:
        cv2.circle(vis, (p["x"], p["y"]), radius, (0, 0, 255), -1)  # red dot (BGR)
    return vis


def overlay_masks(base: np.ndarray, mask_list, alpha: float = 0.4):
    out = base.copy()
    cmap = plt.cm.get_cmap("Set3", len(mask_list))
    for idx, m in enumerate(mask_list):
        color = tuple(int(c * 255) for c in cmap(idx)[:3])[::-1]  # BGR
        layer = np.zeros_like(base)
        layer[m] = color
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    return out

# ─────────────────────────── main ─────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--qwen_dir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--out", default="overlay")  # suffix auto‑added
    args = ap.parse_args()

    # Ensure PNG suffix & current‑dir save
    out_png = Path(f"{Path(args.out).stem}.png").resolve()
    dots_png = Path("cakes_with_points.png").resolve()

    img_path = Path(args.image).expanduser().resolve()
    pil = Image.open(img_path).convert("RGB")
    bgr = cv2.imread(str(img_path))

    proc, qwen = load_qwen(args.qwen_dir)
    points = ask_for_points(proc, qwen, pil)
    print("[Qwen] Detected", len(points), "cakes →", points)

    dots_img = draw_points(np.array(pil), points)
    Image.fromarray(dots_img).save(dots_png)
    print("Dots image saved →", dots_png.name)

    sam = load_sam(args.sam_ckpt)
    masks, scores = segment_masks(sam, bgr, points)
    print("[SAM] IoU per instance:", [f"{s:.3f}" for s in scores])

    overlay = overlay_masks(bgr, masks, alpha=0.4)
    cv2.imwrite(str(out_png), overlay)
    print("Overlay saved →", out_png.name)


if __name__ == "__main__":
    main()
