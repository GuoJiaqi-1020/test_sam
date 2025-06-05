import matplotlib.pyplot as plt

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

def save_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    prefix="",
    ext="png",
):

    for i, (mask, score) in enumerate(zip(masks, scores), 1):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None and input_labels is not None:
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")

        fname = f"{prefix}{i}.{ext}"
        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()
        print("saved:", fname)

# ───────── GPU 可见性 ─────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# ───────── 通用依赖 ─────────
import pathlib, xml.etree.ElementTree as ET, random, re, numpy as np, cv2, torch
from typing import List
from PIL import Image, ImageDraw, ImageColor, ImageFont

# ───────── Qwen (可选) ─────────
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ───────── SAM-2.1 ───────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ========== 配置 ==========
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

# 输出文件
OUT_QWEN = "QWEN_output.png"
OUT_SAM2 = "SAM2_output.png"

# 字体
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

COLORS = list(ImageColor.colormap.keys())

# ---- 跳过 Qwen 相关开关 ----
USE_QWEN     = True          # = False 时走固定/随机点
FIXED_POINTS = [(750, 784, "spoon")]   # 仅当 USE_QWEN=False 且 RANDOM_N=0 时生效
RANDOM_N     = 0             # 设置 >0 则生成随机 N 点


# ───────── 辅助函数 ─────────
def extract_points(xml_text: str) -> List[tuple]:
    """解析 Qwen XML -> [(x,y,label), ...]"""
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


# ========== 主流程 ==========
def main():
    # # ---------- 1. 获取点 ----------
    # if USE_QWEN:
    #     proc  = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #         MODEL_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True
    #     )
    #
    #     messages = [[{
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": f"file://{IMAGE_PATH}"},
    #             {"type": "text",  "text": PROMPT}
    #         ]
    #     }]]
    #     text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     imgs, vids, vkw = process_vision_info(messages, return_video_kwargs=True)
    #     inputs = proc(text=text, images=imgs, videos=vids, return_tensors="pt", **vkw).to(model.device)
    #
    #     with torch.inference_mode():
    #         out_ids = model.generate(**inputs, max_new_tokens=128)
    #     reply = proc.tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    #     print("Qwen reply:\n", reply)
    #     points = extract_points(reply)
    #
    #     if not points:
    #         raise RuntimeError("Qwen 返回为空，检查提示词或模型输出。")
    #
    # else:
    #     # 跳过 LLM
    #     img_w, img_h = Image.open(IMAGE_PATH).size
    #     if RANDOM_N > 0:
    #         points = [(random.randint(0, img_w-1),
    #                    random.randint(0, img_h-1),
    #                    f"rand{i+1}") for i in range(RANDOM_N)]
    #     else:
    #         points = FIXED_POINTS
    #     print("⚡ 使用手动点：", points)
    #
    #     # 构造假的 inputs 以便获取 in_W/in_H
    #     in_W, in_H = img_w, img_h   # 直接用原尺寸
    #     inputs = {"image_grid_thw": np.array([[1, in_H//14, in_W//14]])}
    #
    # # ---------- 2. 绘制 QWEN_output.png ----------
    # img_pil = Image.open(IMAGE_PATH).convert("RGB")
    # W, H = img_pil.size
    # draw = ImageDraw.Draw(img_pil)
    #
    # grid_h, grid_w = inputs["image_grid_thw"][0][1:].tolist()
    # in_H, in_W = grid_h * 14, grid_w * 14
    # font = load_font(20)
    #
    # for i, (x, y, label) in enumerate(points):
    #     vx = x / in_W * W
    #     vy = y / in_H * H
    #     c  = COLORS[i % len(COLORS)]
    #     r  = 10
    #     draw.ellipse([(vx - r, vy - r), (vx + r, vy + r)], fill=c)
    #     draw.text((vx + r + 4, vy + r + 4), label, fill=c, font=font)
    #
    # img_pil.save(OUT_QWEN)
    # print("Points saved →", pathlib.Path(OUT_QWEN).resolve())

    # ---------- 3. SAM-2.1 分割 ----------
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    predictor   = SAM2ImagePredictor(sam2_model)

    orig_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    predictor.set_image(orig_rgb)
    input_point = np.array([[750, 783]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    print('masks shape:', masks.shape)
    save_masks(orig_rgb, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

    # ---------- 4. 叠加 & 保存 ----------
    overlay = orig_rgb.copy()
    alpha   = 0.4
    for idx, m in enumerate(masks):
        color = ImageColor.getrgb(COLORS[idx % len(COLORS)])
        layer = np.zeros_like(orig_rgb)
        layer[m] = color
        overlay = cv2.addWeighted(layer, alpha, overlay, 1 - alpha, 0)

    Image.fromarray(overlay).save(OUT_SAM2)
    print("Overlay saved →", pathlib.Path(OUT_SAM2).resolve())


if __name__ == "__main__":
    main()
