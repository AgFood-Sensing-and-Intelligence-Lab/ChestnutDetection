from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLOv11 model
    model = YOLO('/root/yolov11/model/yolo11m.pt')

    # Train the model with custom settings
    model.train(
        data='/root/yolov11/data.yaml',  # 数据集配置文件路径
        epochs=200,             # 200 轮训练 :contentReference[oaicite:0]{index=0}
        imgsz=1024,             # 输入图像尺寸 1024 :contentReference[oaicite:1]{index=1}
        batch=16,               # 批量大小 16 :contentReference[oaicite:2]{index=2}
        workers=2,              # 数据加载线程数
        project='/root/yolov11/resultsm_3',  # 输出目录
        name='train',           # 训练名称

        optimizer='auto',       # 自动选择优化器 :contentReference[oaicite:3]{index=3}
        lr0=0.01,               # 初始学习率 0.01 :contentReference[oaicite:4]{index=4}
        lrf=0.01, 
        cos_lr=True,            # 余弦学习率调度 :contentReference[oaicite:5]{index=5}
        weight_decay=0.0005,    # 权重衰减 0.0005 :contentReference[oaicite:6]{index=6}
        iou=0.7,                   # NMS IoU 阈值
        warmup_epochs=3.0,      # 预热阶段 3.0 轮 :contentReference[oaicite:7]{index=7}

        amp=True,               # 混合精度训练 (AMP) :contentReference[oaicite:8]{index=8}

        # 数据增强策略：随机水平翻转与缩放
        #fliplr=0.5,             # 随机水平翻转概率 :contentReference[oaicite:9]{index=9}
        #scale=0.5,              # 随机缩放因子 :contentReference[oaicite:10]{index=10}
        augment=True,              # 随机翻转、缩放等自适应增强

        device=[0,1,2,3]
    )
