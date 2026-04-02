#!/usr/bin/env python3
# test.py

from ultralytics import YOLO

def main():

    weights_path = "resultsm_5/exp_auto_opt_cos_lr_amp2/weights/best.pt"
 
    data_yaml    = "data.yaml"

    model = YOLO(weights_path)

    print(f"\n➡️  Evaluating on TEST set\n   weights: {weights_path}\n   data:    {data_yaml}\n")

    metrics = model.val(
        data=data_yaml,
        split="test",
        save_json=True,
        project="runsm_5",    
        name="exp_test"  
    )

    print("=== Test set metrics ===")
    for name, value in (metrics.results_dict() if callable(getattr(metrics, "results_dict", None)) else metrics.results_dict).items():
        print(f"{name:10s}: {value:.4f}")

if __name__ == "__main__":
    main()


