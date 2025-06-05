#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen 点 → SAM-2 分割 → 叠加
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

import re, pathlib, xml.etree.ElementTree as ET
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
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

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

# ---------- 路径 / 提示 ----------
MODEL_DIR   = "./qwen_vl"
IMAGE_PATH  = "./assets/spatial_understanding/cakes.png"
PROMPT      = "Locate the spoon, and output its coordinates in XML format <points x y>object</points>"
SAM2_YAML = "./sam2.1/sam2.1_hiera_b+.yaml"
SAM2_PT   = "./checkpoint/sam2.1_hiera_base_plus.pt"
device    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_QWEN    = "QWEN_output.png"
OUT_SAM2    = "SAM2_output.png"
COLORS      = list(ImageColor.colormap.keys())

# ---------- 加载 Qwen ----------
proc  = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True)

# ---------- 构造消息 ----------
messages = [[{
    "role": "user",
    "content": [
        {"type": "image", "image": f"file://{IMAGE_PATH}"},
        {"type": "text",  "text": PROMPT}
    ]
}]]

# ---------- 编码 ----------
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
inputs = proc(text=text, images=imgs, videos=vids, return_tensors="pt", **vkw).to(model.device)

# ---------- Qwen 推理 ----------
with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=128)
reply = proc.tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("The Prompt:\n", PROMPT)
print("Qwen reply:\n", reply)

# ---------- 解析 XML ----------
def extract_points(xml_text: str):
    xml_text = xml_text.replace("```xml", "").replace("```", "").strip()
    root = ET.fromstring(f"<root>{xml_text}</root>")
    out = []
    for tag in root.findall("points"):
        xs = {k[1:]: int(v) for k, v in tag.attrib.items() if k.startswith("x")}
        ys = {k[1:]: int(v) for k, v in tag.attrib.items() if k.startswith("y")}
        label = tag.attrib.get("alt") or (tag.text.strip() if tag.text else "object")
        for idx in sorted(xs, key=int):
            if idx in ys:
                out.append((xs[idx], ys[idx], label))
    return out

points = extract_points(reply)
if not points:
    raise ValueError("No <points …> annotation parsed!")

# ---------- 画点 PNG ----------
img_pil = Image.open(IMAGE_PATH).convert("RGB")
W, H = img_pil.size
draw = ImageDraw.Draw(img_pil)

grid_h, grid_w = inputs["image_grid_thw"][0][1:].tolist()
in_H, in_W = grid_h * 14, grid_w * 14

DEFAULT_TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    font = ImageFont.truetype(DEFAULT_TTF, size=20)
except IOError:
    font = ImageFont.load_default()

for i, (x, y, label) in enumerate(points):
    vx = x / in_W * W
    vy = y / in_H * H
    c  = COLORS[i % len(COLORS)]
    r  = 10
    draw.ellipse([(vx - r, vy - r), (vx + r, vy + r)], fill=c)
    draw.text((vx + r + 4, vy + r + 4), label, fill=c, font=font)

img_pil.save(OUT_QWEN)
print("Points saved →", pathlib.Path(OUT_QWEN).resolve())
###########################################
###########################################
# ========== SAM-2 分割 ==========
sam2_model   = build_sam2(SAM2_YAML, SAM2_PT, device=device)
predictor    = SAM2ImagePredictor(sam2_model)

orig_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
predictor.set_image(orig_rgb)

print("SAM-2 model loaded:", SAM2_PT)
print("##########################")
print("points to predict:", points)

masks = []
for (x, y, _) in points:
    vx = x / in_W * W
    vy = y / in_H * H
    m, _ = predictor.predict(
        point_coords = np.array([[vx, vy]]),
        point_labels = np.array([1], dtype=np.int32),   # 1 = foreground
        multimask_output = False
    )
    masks.append(m[0])

# ---------- 半透明叠加 ----------
overlay = orig_rgb.copy()
alpha   = 0.4
for idx, m in enumerate(masks):
    color = ImageColor.getrgb(COLORS[idx % len(COLORS)])
    layer = np.zeros_like(orig_rgb)
    layer[m] = color
    overlay = cv2.addWeighted(layer, alpha, overlay, 1 - alpha, 0)

Image.fromarray(overlay).save("SAM2_output.png")
print("Overlay saved →", pathlib.Path("SAM2_output.png").resolve())