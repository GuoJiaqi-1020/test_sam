#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qwen_xml_points_demo.py
-----------------------
输入图片 -> Qwen 2.5 VL 输出 XML 坐标 -> 画点。

准备：
conda activate qwen_sam2   # 你的环境
pip install git+https://github.com/QwenLM/Qwen-VL.git
# 权重放到 ./qwen_vl
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import re, json, random, pathlib, ast, os
from typing import List
from PIL import Image, ImageDraw, ImageFont, ImageColor
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ---------- 配置 ----------
MODEL_DIR  = "./qwen_vl"
IMAGE_PATH = "./assets/spatial_understanding/cakes.png"
# PROMPT = "point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>"
PROMPT = "Locate the spoon, and output its coordinates in XML format <points x y>object</points>"
OUT_PNG = "Qwen_output.png"
COLORS = list(ImageColor.colormap.keys())

# ---------- 加载模型 ----------
proc  = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True)

# ---------- 构造 messages ----------
messages = [[{
    "role": "user",
    "content": [
        {"type": "image", "image": f"file://{IMAGE_PATH}"},
        {"type": "text",  "text": PROMPT}
    ]
}]]

# ---------- Processor 打包 ----------
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
inputs = proc(text=text, images=imgs, videos=vids, return_tensors="pt", **vkw).to(model.device)

# ---------- 推理 ----------
with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=128)
reply = proc.tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("The Prompt:\n", PROMPT)
print("Qwen reply:\n", reply)

# ---------- 解析 XML ----------
import xml.etree.ElementTree as ET

def extract_points(xml_text: str):
    xml_text = xml_text.replace("```xml", "").replace("```", "").strip()
    root = ET.fromstring(f"<root>{xml_text}</root>")
    out = []

    for tag in root.findall("points"):
        # ---------- 拿坐标 ----------
        xs = {k[1:]: int(v) for k, v in tag.attrib.items() if k.startswith("x")}
        ys = {k[1:]: int(v) for k, v in tag.attrib.items() if k.startswith("y")}
        # ---------- 拿文字 ----------
        label = tag.attrib.get("alt") or (tag.text.strip() if tag.text else "object")

        for idx in sorted(xs, key=int):
            if idx in ys:
                out.append((xs[idx], ys[idx], label))
    return out

points = extract_points(reply)
if not points:
    raise ValueError("No <points …> annotation parsed!")

# ---------- 绘图 ----------
img = Image.open(IMAGE_PATH).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)

grid_h, grid_w = inputs["image_grid_thw"][0][1:].tolist()
in_H, in_W = grid_h * 14, grid_w * 14
print(f"Image size: {W}x{H}, Grid size sent to model: {in_W}x{in_H}")

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

img.save(OUT_PNG)
print("Saved →", pathlib.Path(OUT_PNG).resolve())