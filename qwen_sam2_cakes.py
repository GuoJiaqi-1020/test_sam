"""qwen_sam2_cakes.py
====================
End‑to‑end demo:
1. Load image (default `./assets/spatial_understanding/cakes.png`).
2. Use **Qwen‑2.5‑VL** to predict center‑points for every cake.
3. Draw red dots → `cakes_with_points.png`.
4. Feed points to **SAM (Vit‑H)** → masks.
5. Overlay masks with 40 % alpha → `<out>.png` (default `overlay.png`).

### NEW: GPU selection **inside the script**
Pass `--gpus 0,1,2,3` (or any comma list). The script sets
`CUDA_VISIBLE_DEVICES` *before* importing PyTorch, so no external env tweak
is required.

```bash
python qwen_sam2_cakes.py \
  --gpus 0,1,2,3 \
  --image ./assets/spatial_understanding/cakes.png \
  --qwen_dir ./qwen_vl \
  --sam_ckpt ./sam_ckpt/sam_vit_h_4b8939.pth \
  --out overlay
```
"""

# ─────────────────────── early GPU parsing ────────────────────────
import os, sys
if "--gpus" in sys.argv:
    i = sys.argv.index("--gpus")
    if i + 1 >= len(sys.argv):
        raise RuntimeError("--gpus requires a value, e.g. --gpus 0,1,2")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
    # remove the two tokens so argparse later won't complain
    sys.argv.pop(i + 1)
    sys.argv.pop(i)

# ───────────────────────── imports ────────────────────────────────
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

# ───────────────────────── helpers ────────────────────────────────

def load_qwen(model_dir: str, device: str = "cuda"):
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    return proc, model


def ask_for_points(proc, model, pil_img: Image.Image) -> List[Dict[str, int]]:
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
    return [{"x": int(p["x"]), "y": int(p["y"])} for p in pts]


def load_sam(ckpt: str, model_type="vit_h"):
    return SamPredictor(checkpoint=ckpt, model_type=model_type)


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


def draw_points(img: np.ndarray, pts, r=6):
    vis = img.copy()
    for p in pts:
        cv2.circle(vis, (p["x"], p["y"]), r, (0, 0, 255), -1)
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

# ─────────────────────────── main ────────────────────────────────

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

    sam = load_sam(args.sam_ckpt)
    masks, sc = segment_masks(sam, bgr, pts)
    print("[SAM] IoU:", [f"{s:.3f}" for s in sc])

    cv2.imwrite(str(out_png), overlay_masks(bgr, masks))
    print("Overlay saved →", out_png.name)


if __name__ == "__main__":
    main()
