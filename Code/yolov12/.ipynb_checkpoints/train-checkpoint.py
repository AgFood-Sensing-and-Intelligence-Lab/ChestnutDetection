#!/usr/bin/env python3
# train.py

import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv12 with custom hyperparameters")
    parser.add_argument("--data",    type=str, default="data.yaml", help="path to data.yaml")
    parser.add_argument("--model",   type=str, default="yolov12x.yaml",    help="model config (n/s/m/l/x)")
    parser.add_argument("--epochs",  type=int, default=200,                help="number of training epochs")
    parser.add_argument("--batch",   type=int, default=16,                 help="batch size")
    parser.add_argument("--imgsz",   type=int, default=1024,               help="input image size")
    parser.add_argument("--device",  type=str, default="0,1,2,3",                help="GPU device")
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载本地覆写后的 YOLOv12 实现
    model = YOLO(args.model)

    # 启动训练
    results = model.train(
        data=args.data,            # 数据配置
        epochs=args.epochs,        # 训练轮数：200
        batch=args.batch,          # 批量大小：16
        imgsz=args.imgsz,          # 输入尺寸：1024
        device=args.device,        # GPU 设备

        # 优化器与学习率
        optimizer="auto",          # 自动选择最优优化器
        lr0=0.0005,                  # 初始学习率
        lrf=0.01,                  # 余弦退火终点 lr = lr0 * lrf
        cos_lr=True,               # 启用余弦学习率调度
        warmup_epochs=3.0,         # 3 轮预热

        # 正则化与阈值
        weight_decay=0.0005,       # 权重衰减
        iou=0.7,                   # NMS IoU 阈值

        # 训练加速与增强
        amp=True,                  # 自动混合精度（AMP）
        augment=True,              # 随机翻转、缩放等自适应增强

        # 输出配置
        save_json=True,            # 每张图预测保存 JSON 格式
        project="resultsx_5",         # 输出目录
        name="exp_auto_opt_cos_lr_amp"  # 本次实验名称
    )

    print("Training completed.")
    print(results)  # 打印训练日志与最终指标

if __name__ == "__main__":
    main()


