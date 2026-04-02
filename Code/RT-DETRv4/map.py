import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

import sys
# 让 Python 找到工程路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))

# 根据仓库结构导入模型构建函数
from models.rt_detr import build_rt_detrv4

# 全局保存特征图的字典
feature_maps = {}

def save_feature(name):
    """hook 记录指定层的输出特征图"""
    def hook(module, inp, out):
        feature_maps[name] = out.detach().cpu()
    return hook

def visualize_and_save(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, fmap in feature_maps.items():
        # 取 batch 中第一个样本并对通道平均，得到单通道图
        fmap_avg = fmap[0].mean(dim=0)
        plt.figure(figsize=(5,5))
        plt.imshow(fmap_avg.numpy(), cmap='viridis')
        plt.title(f'Feature Map {name}')
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, f'{name}.png'))
        plt.close()

def main():
    # 1) 加载模型
    model, _, _ = build_rt_detrv4()
    model.eval()

    # 2) 注册 hook 到 backbone 多尺度输出
    # 这些是主干网络的多尺度卷积输出
    model.backbone.layer2.register_forward_hook(save_feature('S3'))
    model.backbone.layer3.register_forward_hook(save_feature('S4'))
    model.backbone.layer4.register_forward_hook(save_feature('S5'))

    # 3) 图像预处理
    img_path = "/root/autodl-tmp/data/test_rotated/images/20241005_NashFarm_iPhone12_YL_101.jpg"
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    inp = transform(img).unsqueeze(0)  # 1xCxHxW

    # 4) 推理（提取特征）
    with torch.no_grad():
        _ = model(inp)

    # 5) 保存/可视化特征图
    visualize_and_save(out_dir="./rtdetrv4_feats")

if __name__ == "__main__":
    main()