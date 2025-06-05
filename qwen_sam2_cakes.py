"""qwen_sam2_cakes.py
====================
End‑to‑end demo that
1. loads `./assets/spatial_understanding/cakes.png`,
2. asks Qwen‑2.5‑VL to output the (x, y) coordinates of **all** cakes in the picture,
3. plots those points on the original image (red dots),
4. feeds each point to SAM 2 to obtain an instance mask, and
5. overlays every mask on the original image with 40 % transparency.

Usage:
    # (after installing requirements and downloading checkpoints)
    python qwen_sam2_cakes.py \
        --image ./assets/spatial_understanding/cakes.png \
        --qwen_dir ./qwen_vl \
        --sam_ckpt ./sam2_ckpt/sam_vit_h_4b8939.pth \
        --out overlay.png

The script saves two files:
    * cakes_with_points.png – dots only
    * <out>              – image with semi‑transparent masks
"""

import argparse, json, os, re, tempfile
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq
from sam2.modeling import SamPredictor

# ----------------------------- Qwen helpers ----------------------------- #

def load_qwen(model_dir: str, device: str = "cuda"):
    """Load processor + VL model in FP16."""
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    return processor, model


def ask_for_points(processor, model, pil_img: Image.Image) -> List[Dict[str, int]]:
    """Prompt Qwen to return every cake center as JSON list."""
    prompt = (
        "Identify the approximate center point of **each** cake in the picture. "
        "Return a JSON list where every element is an object with integer 'x' and 'y' keys "
        "representing pixel coordinates in the *original resolution*. Example: "
        "[{\"x\": 123, \"y\": 456}, ...]."
    )

    inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=128)
    txt = processor.tokenizer.decode(gen[0], skip_special_tokens=True)

    # basic cleanup – strip anything before/after the first [
    m = re.search(r"\[[\s\S]*\]", txt)
    if not m:
        raise ValueError(f"Qwen did not return JSON list, got: {txt}")
    pts = json.loads(m.group(0))
    if not isinstance(pts, list):
        raise ValueError("Parsed JSON is not a list")
    return [{"x": int(p["x"]), "y": int(p["y"])} for p in pts]


# ----------------------------- SAM‑2 helpers ----------------------------- #

def load_sam2(ckpt: str):
    predictor = SamPredictor(checkpoint=ckpt, model_type="vit_h")
    return predictor


def segment_masks(predictor, img_bgr: np.ndarray, points: List[Dict[str, int]]):
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    all_masks, all_scores = [], []
    for pt in points:
        coord = np.array([[pt["x"], pt["y"]]])
        labels = np.array([1], dtype=np.int32)  # foreground
        masks, scores, _ = predictor.predict(point_coords=coord, point_labels=labels, multimask_output=False)
        all_masks.append(masks[0])
        all_scores.append(float(scores[0]))
    return all_masks, all_scores


# ----------------------------- visualisation ----------------------------- #

def draw_points(pil_img: Image.Image, points: List[Dict[str, int]], out_file: Path):
    img = np.array(pil_img).copy()
    for p in points:
        cv2.circle(img, (p["x"], p["y"]), radius=6, color=(255, 0, 0), thickness=-1)
    Image.fromarray(img).save(out_file)
    return img


def overlay_masks(base: np.ndarray, masks: List[np.ndarray], alpha: float = 0.4):
    overlay = base.copy()
    colour_table = plt.cm.get_cmap("Set3", len(masks))
    for idx, m in enumerate(masks):
        colour = tuple(int(c * 255) for c in colour_table(idx)[:3])  # RGB
        coloured = np.zeros_like(base)
        coloured[m] = colour[::-1]  # BGR order for cv2
        overlay = cv2.addWeighted(coloured, alpha, overlay, 1 - alpha, 0)
    return overlay


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--qwen_dir", required=True, help="local dir containing Qwen‑2.5‑VL weights")
    ap.add_argument("--sam_ckpt", required=True, help="SAM‑2 vit_h checkpoint path")
    ap.add_argument("--out", default="overlay.png", help="filename for final overlay")
    args = ap.parse_args()

    img_path = Path(args.image)
    pil = Image.open(img_path).convert("RGB")
    base_bgr = cv2.imread(str(img_path))

    # 1. Qwen → points
    proc, qwen = load_qwen(args.qwen_dir)
    points = ask_for_points(proc, qwen, pil)
    print("Detected", len(points), "cakes →", points)

    # 2. draw dots & save
    draw_points(pil, points, img_path.with_name("cakes_with_points.png"))

    # 3. SAM‑2 → masks
    predictor = load_sam2(args.sam_ckpt)
    masks, scores = segment_masks(predictor, base_bgr, points)
    print("SAM2 IoU per instance:", [f"{s:.3f}" for s in scores])

    # 4. overlay masks
    overlay = overlay_masks(base_bgr, masks, alpha=0.4)
    cv2.imwrite(args.out, overlay)
    print("Overlay saved to", args.out)


if __name__ == "__main__":
    main()
