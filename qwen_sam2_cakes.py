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
PROMPT = (
    "在图片中以点形式标出每个小蛋糕中心，并用 XML 输出："
    "<points x1 y1>cake</points><points x2 y2>cake</points>..."
)
OUT_PNG = "cakes_with_points.png"
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
print("Qwen reply:\n", reply)

# ---------- 解析 XML ----------
pts: List[List[int]] = []
for x, y in re.findall(r"<points\\s+(\\d+)\\s+(\\d+)>", reply):
    pts.append([int(x), int(y)])

if not pts:
    raise ValueError("未解析出任何 <points x y> 标注！")

# ---------- 绘图 ----------
img = Image.open(IMAGE_PATH).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)

# Qwen 输入前对图片做了缩放，需映射回原分辨率
grid_h, grid_w = inputs["image_grid_thw"][0][1:].tolist()  # patch 格子数
in_H, in_W = grid_h * 14, grid_w * 14                      # 每 patch 14 px

for i, (x, y) in enumerate(pts):
    vis_x = x / in_W * W
    vis_y = y / in_H * H
    c = COLORS[i % len(COLORS)]
    r = 6
    draw.ellipse([(vis_x - r, vis_y - r), (vis_x + r, vis_y + r)], fill=c)
    draw.text((vis_x + r + 4, vis_y + r + 4), f"cake{i+1}", fill=c)

img.save(OUT_PNG)
print("Saved →", pathlib.Path(OUT_PNG).resolve())
