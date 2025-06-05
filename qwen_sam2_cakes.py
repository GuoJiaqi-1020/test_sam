import matplotlib.pyplot as plt
import numpy as np

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def save_masks(
    image: np.ndarray,
    masks: list,
    scores: list,
    point_coords: np.ndarray = None,
    box_coords: np.ndarray = None,
    input_labels: np.ndarray = None,
    borders: bool = True,
    prefix: str = "",
    ext: str = "png",
):
    """
    Save multiple masks sequentially as prefix1.png, prefix2.png, ... in the current directory.
    Supports visualizing points (point_coords, input_labels) and boxes (box_coords).

    Args:
        image:          Original RGB image, shape=(H, W, 3), dtype=uint8.
        masks:          List of masks, each is a float or bool array of shape=(h, w).
        scores:         Confidence score list corresponding to the masks.
        point_coords:   Optional, shape=(N,2) point coordinates for overlay.
        box_coords:     Optional, box coordinates in the format (x0, y0, x1, y1).
        input_labels:   Optional, shape=(N,) array of 1/0 indicating positive/negative points.
        borders:        Whether to draw contours around the mask (default: True).
        prefix:         Filename prefix for saving (e.g., "sam_" → sam_1.png).
        ext:            File extension (default: "png").

    Notes:
        - If mask resolution ≠ original image, it will be resized using INTER_NEAREST.
        - If mask dtype is not bool, it will be converted with (mask > 0.5).
    """
    H, W = image.shape[:2]

    def show_mask(mask, ax, random_color=False, borders=True):
        """Draw a single mask onto matplotlib axis."""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])  # RGBA
        h, w = mask.shape[-2:]
        mask_uint = (mask.astype(np.uint8) if mask.dtype != np.uint8 else mask)
        mask_image = mask_uint.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(cnt, epsilon=0.01, closed=True) for cnt in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos = coords[labels == 1]
        neg = coords[labels == 0]
        ax.scatter(pos[:, 0], pos[:, 1], color="green", marker="*", s=marker_size,
                   edgecolor="white", linewidth=1.25)
        ax.scatter(neg[:, 0], neg[:, 1], color="red",   marker="*", s=marker_size,
                   edgecolor="white", linewidth=1.25)

    def show_box(box, ax):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green",
                                   facecolor=(0, 0, 0, 0), lw=2))

    for i, (mask, score) in enumerate(zip(masks, scores), start=1):
        m_bool = (mask > 0.5) if mask.dtype != np.bool_ else mask

        if m_bool.shape != (H, W):
            m_uint = m_bool.astype(np.uint8)
            m_resized = cv2.resize(m_uint, (W, H), interpolation=cv2.INTER_NEAREST)
            m_bool = m_resized.astype(bool)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        show_mask(m_bool, ax, borders=borders)

        if point_coords is not None and input_labels is not None:
            show_points(point_coords, input_labels, ax)

        if box_coords is not None:
            show_box(box_coords, ax)

        if len(scores) > 1:
            ax.set_title(f"Mask {i}, Score: {score:.3f}", fontsize=16)

        ax.axis("off")
        fname = f"{prefix}{i}.{ext}"
        fig.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print("saved:", fname)

# ───────── GPU visibility ─────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# ───────── Common dependencies ─────────
import pathlib, xml.etree.ElementTree as ET, random, re, numpy as np, cv2, torch
from typing import List
from PIL import Image, ImageDraw, ImageColor, ImageFont

# ───────── Qwen (optional) ─────────
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ───────── SAM-2.1 ─────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ========== Configurations ==========
IMAGE_PATH  = "./assets/spatial_understanding/cakes.png"
MODEL_DIR   = "./qwen_vl"
PROMPT      = (
    "Locate the spoon, and output its coordinates in XML format "
    "<points x y>object</points>"
)

# SAM-2.1
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Output files
OUT_QWEN = "QWEN_output.png"
OUT_SAM2 = "SAM2_output.png"

# Font
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

COLORS = list(ImageColor.colormap.keys())

# ---- Qwen bypass toggle ----
USE_QWEN     = True          # = False to use fixed/random points
FIXED_POINTS = [(750, 784, "spoon")]   # Used only if USE_QWEN=False and RANDOM_N=0
RANDOM_N     = 0             # Set >0 to randomly generate N points

# ───────── Utility functions ─────────
def extract_points(xml_text: str) -> List[tuple]:
    """Parse Qwen XML -> [(x,y,label), ...]"""
    xml_text = xml_text.replace("```xml", "").replace("```", "").strip()
    root = ET.fromstring(f"<root>{xml_text}</root>")
    outs = []
    for tag in root.findall("points"):
        xs = {k[1:]: int(v) for k, v in tag.attrib.items() if k.startswith("x")}
        ys = {k[1:]: int(v) for k, v in tag.attrib.items() if k.startswith("y")}
        label = tag.attrib.get("alt") or (tag.text.strip() if tag.text else "object")
        for idx in sorted(xs, key=int):
            if idx in ys:
                outs.append((xs[idx], ys[idx], label))
    return outs

def load_font(size=20):
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except IOError:
        return ImageFont.load_default()

# ========== Main workflow ==========
def main():
    # ---------- 1. Get point(s) ----------
    if USE_QWEN:
        proc  = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{IMAGE_PATH}"},
                {"type": "text",  "text": PROMPT}
            ]
        }]]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
        inputs = proc(text=text, images=imgs, videos=vids, return_tensors="pt", **vkw).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=128)
        reply = proc.tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print("Qwen reply:\n", reply)
        points = extract_points(reply)

        if not points:
            raise RuntimeError("Qwen output is empty. Check the prompt or model output.")
    else:
        # Skip LLM
        img_w, img_h = Image.open(IMAGE_PATH).size
        if RANDOM_N > 0:
            points = [(random.randint(0, img_w-1),
                       random.randint(0, img_h-1),
                       f"rand{i+1}") for i in range(RANDOM_N)]
        else:
            points = FIXED_POINTS
        print("⚡ Using manual points:", points)

        in_W, in_H = img_w, img_h   # Use original dimensions
        inputs = {"image_grid_thw": np.array([[1, in_H//14, in_W//14]])}

    # ---------- 2. Draw QWEN_output.png ----------
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    W, H = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    grid_h, grid_w = inputs["image_grid_thw"][0][1:].tolist()
    in_H, in_W = grid_h * 14, grid_w * 14
    font = load_font(20)

    for i, (x, y, label) in enumerate(points):
        vx = x / in_W * W
        vy = y / in_H * H
        c  = COLORS[i % len(COLORS)]
        r  = 10
        draw.ellipse([(vx - r, vy - r), (vx + r, vy + r)], fill=c)
        draw.text((vx + r + 4, vy + r + 4), label, fill=c, font=font)

    img_pil.save(OUT_QWEN)
    print("Points saved →", pathlib.Path(OUT_QWEN).resolve())

    # ---------- 3. SAM-2.1 segmentation ----------
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    predictor   = SAM2ImagePredictor(sam2_model)

    orig_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    predictor.set_image(orig_rgb)

    input_label = np.array([1])
    pts_xy = np.array([(x, y) for x, y, _ in points], dtype=np.float32)
    masks, scores, logits = predictor.predict(
        point_coords=pts_xy,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    print('masks shape:', masks.shape)
    save_masks(orig_rgb, masks, scores, point_coords=pts_xy, input_labels=input_label, borders=True)
    print("SAM2 masks saved !!!")

if __name__ == "__main__":
    main()
