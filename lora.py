from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch
import os
from PIL import Image
import requests
from io import BytesIO

print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"PyTorch 版本: {torch.__version__}")


def apply_lora_weights(unet, lora_path):
    """加载 LoRA 权重并应用到 UNet 模型"""
    try:
        # 确保 lora_path 文件存在
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA 权重文件 {lora_path} 未找到！")

        # 使用 safetensors 加载 LoRA 权重
        state_dict = load_file(lora_path)
        for name, param in unet.named_parameters():
            if name in state_dict:
                # 将 LoRA 权重应用到 UNet 参数
                param.data += state_dict[name].to(param.device)
                print(f"应用 LoRA 权重到 {name}")
    except Exception as e:
        print(f"应用 LoRA 权重时出错: {e}")


# 加载基础模型
base_model_path = r""
try:
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"基础模型文件 {base_model_path} 未找到！")

    # 加载模型并转移到 CUDA 设备
    pipeline = StableDiffusionPipeline.from_single_file(base_model_path, torch_dtype=torch.float16)
    pipeline.to("cuda")
    print("成功加载模型")
except Exception as e:
    print(f"加载 Stable Diffusion 模型时出错: {e}")
    exit(1)

# 加载 LoRA 权重并应用
lora_path = r"D:\sd-webui-aki-v4\models\Lora\CNhuawen.safetensors"
apply_lora_weights(pipeline.unet, lora_path)


# 加载风格参考图像，用于风格迁移
def load_image(image_path):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    return img

# 文件路径
style_image_path = ""
style_image = load_image(style_image_path)

style_image = style_image.resize((512, 512))

prompt = (
    "A green and purple intricate and elegant floral pattern, featuring delicate blooming flowers, swirling vines, and soft pastel "
    "colors like pink, lavender, and light blue. The design should have a symmetrical and balanced composition, "
    "with fine details and a sense of harmony. The pattern should resemble traditional Chinese floral motifs with an "
    "elegant, timeless feel, incorporating soft gradients and subtle textures."
)
try:
    init_image = style_image
    init_image = init_image.convert("RGB").resize((512, 512))

    # 风格迁移的强度（0 到 1 之间）
    strength = 0.5
    # 文生图
    # image = pipeline(prompt=prompt, strength=strength).images[0]
    # 图生图
    image = pipeline(prompt=prompt, init_image=init_image, strength=strength).images[0]

    # 保存生成的图像
    image.save("output_styled_image.png")
    print("图片已保存为 output_styled_image.png")
except Exception as e:
    print(f"生成图片时出错: {e}")
