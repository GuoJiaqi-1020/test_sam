from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("./qwen_vl", trust_remote_code=True)

img_token      = getattr(proc.tokenizer, "image_token", None)
img_token_id   = getattr(proc.tokenizer, "image_token_id", None)
print("token literal:", img_token)
print("token id     :", img_token_id)

demo = f"{img_token} please describe the picture."
ids  = proc.tokenizer(demo).input_ids
print("count of image tokens in ids:", ids.count(img_token_id))