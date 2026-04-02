#!/usr/bin/env python3
# test.py

from ultralytics import YOLO

def main():
    # 权重文件
    weights_path = "resultsx_5/exp_auto_opt_cos_lr_amp2/weights/best.pt"
    # 数据配置
    data_yaml    = "data.yaml"

    # 加载模型
    model = YOLO(weights_path)

    print(f"\n➡️  Evaluating on TEST set\n   weights: {weights_path}\n   data:    {data_yaml}\n")

    # 评估 test 集，并把 JSON 保存到 runsn/exp*/labels_json
    metrics = model.val(
        data=data_yaml,
        split="test",
        save_json=True,
        project="runsx_5",    # <- 这里指定输出到 runsn 而不是 runs/val
        name="exp_test"     # <- 可选：在 runsn 下的子文件夹名称
    )

    # 打印指标
    print("=== Test set metrics ===")
    for name, value in (metrics.results_dict() if callable(getattr(metrics, "results_dict", None)) else metrics.results_dict).items():
        print(f"{name:10s}: {value:.4f}")

if __name__ == "__main__":
    main()


