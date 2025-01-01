# RobotDesign
用conda创建虚拟环境

```
conda create -n your_env_name python=3.9  # your_env_name 为你虚拟环境名
```

激活虚拟环境

```
conda activate your_env_name
```

安装环境\或者按照自己的驱动版本安装

```
pip install -r requirements.txt
```

在代码中填入模型和路径信息

```python
# 基础模型
base_model_path = r"path_to_your_model"
# lora
lora_path = r"path_to_your_lora"
# 图片路径
style_image_path = "path_to_your_image"
```

选择文生图还是图生图

```python
# 文生图
# image = pipeline(prompt=prompt, strength=strength).images[0]
# 图生图
image = pipeline(prompt=prompt, init_image=init_image, strength=strength).images[0]
```

