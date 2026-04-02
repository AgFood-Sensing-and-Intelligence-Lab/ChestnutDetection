#!/usr/bin/env python3
import os
import random
import shutil
import yaml

# 路径设置
BASE_DIR = "/root/autodl-tmp/data825"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
OUT_DIR = BASE_DIR  # 输出还是在 data825 目录下

# 划分数量
NUM_TRAIN = 223
NUM_VAL   = 31
NUM_TEST  = 65
TOTAL = NUM_TRAIN + NUM_VAL + NUM_TEST

# 确保输出目录存在
os.makedirs(OUT_DIR, exist_ok=True)

# 获取所有图片（假设 labels 同名 .txt）
all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
assert len(all_images) >= TOTAL, f"图片不足 {TOTAL} 张，实际只有 {len(all_images)} 张"

for split_id in range(1, 6):  # 生成5次
    print(f"\n========== 生成第 {split_id} 次划分 ==========")

    # 随机打乱
    files = all_images[:]
    random.shuffle(files)

    train_files = files[:NUM_TRAIN]
    val_files   = files[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
    test_files  = files[NUM_TRAIN+NUM_VAL:NUM_TRAIN+NUM_VAL+NUM_TEST]

    # 输出目录
    split_dir = os.path.join(OUT_DIR, str(split_id))
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

    # 拷贝函数
    def copy_files(file_list, subset):
        for f in file_list:
            img_src = os.path.join(IMG_DIR, f)
            lbl_src = os.path.join(LBL_DIR, os.path.splitext(f)[0] + ".txt")

            img_dst = os.path.join(split_dir, "images", subset, f)
            lbl_dst = os.path.join(split_dir, "labels", subset, os.path.splitext(f)[0] + ".txt")

            shutil.copy(img_src, img_dst)
            if os.path.exists(lbl_src):  # 标签可能缺失，跳过
                shutil.copy(lbl_src, lbl_dst)

    # 拷贝文件
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    # 写 data.yaml
    data_yaml = {
        "train": os.path.join(split_dir, "images/train"),
        "val": os.path.join(split_dir, "images/val"),
        "test": os.path.join(split_dir, "images/test"),
        "nc": 1,
        "names": ["0"]
    }
    with open(os.path.join(split_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ 第 {split_id} 次划分完成，结果存放在 {split_dir}")