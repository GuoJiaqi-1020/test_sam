from PIL import Image
from qwen_vl_utils import inference, plot_points   # 就在官方 repo 根目录 / qwen_vl_utils.py

# ------------------------------------------------------------
image_path = "./assets/spatial_understanding/cakes.png"

cn_prompt = "以点的形式定位图中桌子远处的擀面杖，以XML格式输出其坐标"
en_prompt = (
    "point to the rolling pin on the far side of the table, "
    "output its coordinates in XML format <points x y>object</points>"
)
prompt = en_prompt        # 二选一

# 2. 推理：函数内部自动插入 <img> 占位符
response, in_h, in_w = inference(image_path, prompt, model_dir="./qwen_vl")
# ------------------------------------------------------------

# 3. 可视化
image = Image.open(image_path)
image.thumbnail([640, 640], Image.Resampling.LANCZOS)
plot_points(image, response, in_w, in_h)   # 会直接在窗口里弹出 / Jupyter 中渲染
image.save("cakes_with_points.png")     