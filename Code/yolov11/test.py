#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Precision-Recall curve points from Ultralytics YOLO validation results.

Output:
    <save_dir>/pr_curve_points.csv

CSV format:
    recall,precision
    0.000000,1.000000
    ...

Usage:
    python test_pr_export.py \
        --weights /root/yolov11/resultsx_5/train2/weights/best.pt \
        --data /root/yolov11/data.yaml \
        --imgsz 1024 \
        --conf 0.001 \
        --iou 0.7 \
        --project runsx_5 \
        --name exp_test
"""

import os
import csv
import argparse
from typing import Any, Iterable

from ultralytics import YOLO


def to_list(x: Any):
    """Safely convert tensor/ndarray/list-like objects to Python list."""
    if x is None:
        return None
    # torch.Tensor / numpy.ndarray may have .tolist()
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    # generic iterable
    try:
        return list(x)
    except Exception:
        return None


def flatten_if_single_class(y):
    """
    Ultralytics curve y may be:
      - 1D list: [p1, p2, ...]
      - 2D list for multi-class: [[...], [...], ...]
    For single-class detection, convert [[...]] -> [...]
    For multi-class, this script exports the mean curve across classes.
    """
    if not isinstance(y, list) or len(y) == 0:
        return y

    # 2D case
    if isinstance(y[0], list):
        if len(y) == 1:
            return y[0]
        # multi-class: mean across classes
        length = min(len(row) for row in y if isinstance(row, list) and len(row) > 0)
        if length == 0:
            return []
        mean_curve = []
        for i in range(length):
            vals = []
            for row in y:
                if isinstance(row, list) and len(row) > i:
                    vals.append(float(row[i]))
            mean_curve.append(sum(vals) / len(vals))
        return mean_curve

    return y


def save_pr_csv(save_path: str, recall, precision):
    """Save PR points to CSV."""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recall", "precision"])
        for r, p in zip(recall, precision):
            writer.writerow([f"{float(r):.6f}", f"{float(p):.6f}"])


def main(opt):
    model = YOLO(opt.weights)

    print(f"\n➡️ Evaluating on TEST set")
    print(f"   weights: {opt.weights}")
    print(f"   data:    {opt.data}")
    print(f"   imgsz:   {opt.imgsz}")
    print(f"   conf:    {opt.conf}")
    print(f"   iou:     {opt.iou}\n")

    metrics = model.val(
        data=opt.data,
        split="test",
        imgsz=opt.imgsz,
        conf=opt.conf,       # 推荐 0.001，用于计算PR曲线
        iou=opt.iou,
        save_json=True,
        project=opt.project,
        name=opt.name
    )

    # 打印主要指标
    print("=== Test set metrics ===")
    results_dict = metrics.results_dict() if callable(getattr(metrics, "results_dict", None)) else metrics.results_dict
    for k, v in results_dict.items():
        try:
            print(f"{k:20s}: {float(v):.6f}")
        except Exception:
            print(f"{k:20s}: {v}")

    # 输出目录
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is None:
        save_dir = os.path.join(opt.project, opt.name)
    os.makedirs(save_dir, exist_ok=True)

    # 读取曲线
    curve_names = getattr(metrics, "curves", [])
    curve_results = getattr(metrics, "curves_results", [])

    print("\n=== Available curves ===")
    for i, name in enumerate(curve_names):
        print(f"[{i}] {name}")

    # 找 Precision-Recall(B)
    pr_index = None
    for i, name in enumerate(curve_names):
        if "precision-recall" in str(name).lower():
            pr_index = i
            break

    if pr_index is None:
        raise RuntimeError(
            "No Precision-Recall curve found in metrics.curves. "
            "Please print metrics.curves and check your ultralytics version."
        )

    pr_data = curve_results[pr_index]

    """
    Ultralytics curves_results 常见结构类似：
        [x, y, x_title, y_title]
    对于 PR 曲线通常：
        x = recall
        y = precision
    但不同版本可能存在轻微差异，因此这里做稳健解析。
    """
    if not isinstance(pr_data, (list, tuple)) or len(pr_data) < 2:
        raise RuntimeError(f"Unexpected PR curve data format: {type(pr_data)}")

    x = to_list(pr_data[0])   # expected recall
    y = to_list(pr_data[1])   # expected precision

    if x is None or y is None:
        raise RuntimeError("Failed to convert PR curve data to list.")

    y = flatten_if_single_class(y)

    # 兼容某些版本 y 仍可能不是1D
    if isinstance(y, list) and len(y) > 0 and isinstance(y[0], list):
        raise RuntimeError("PR curve precision data is still 2D after processing; please inspect pr_data manually.")

    if len(x) == 0 or len(y) == 0:
        raise RuntimeError("PR curve data is empty.")

    n = min(len(x), len(y))
    recall = x[:n]
    precision = y[:n]

    save_path = os.path.join(save_dir, "pr_curve_points.csv")
    save_pr_csv(save_path, recall, precision)

    print(f"\n✅ PR curve points saved to: {save_path}")
    print("CSV columns: recall, precision")
    print("You can use this file directly to plot the Precision-Recall curve.")


def parse_opt():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on test set and export PR curve CSV")
    parser.add_argument(
        "--weights",
        type=str,
        default="/root/yolov11/resultsx_5/train2/weights/best.pt",
        help="模型权重路径"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/root/yolov11/data.yaml",
        help="数据集配置文件路径"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="输入图像尺寸"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="置信度阈值。导出PR曲线建议设为0.001"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU阈值"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runsx_5",
        help="结果输出根目录"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp_test",
        help="输出子文件夹名称"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)