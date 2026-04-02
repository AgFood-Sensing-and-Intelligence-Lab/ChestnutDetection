#!/usr/bin/env python3
# train.py

import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv13 with custom hyperparameters")
    parser.add_argument("--data",    type=str, default="data.yaml", help="path to data.yaml")
    parser.add_argument("--model",   type=str, default="yolov13m.yaml",    help="model config (n/s/m/l/x)")
    parser.add_argument("--epochs",  type=int, default=200,                help="number of training epochs")
    parser.add_argument("--batch",   type=int, default=16,                 help="batch size")
    parser.add_argument("--imgsz",   type=int, default=1024,               help="input image size")
    parser.add_argument("--device",  type=str, default="0,1,2,3,4,5,6,7",                help="GPU device")
    return parser.parse_args()

def main():
    args = parse_args()


    model = YOLO(args.model)


    results = model.train(
        data=args.data,           
        epochs=args.epochs,      
        batch=args.batch,        
        imgsz=args.imgsz,        
        device=args.device,       

        optimizer="AdamW",        
        lr0=0.0005,              
        lrf=0.01,
        momentum=0.9,
        cos_lr=True,             
        warmup_epochs=3.0,        

        weight_decay=0.0005,      
        iou=0.7,                   


        amp=True,                 
        augment=True,            

        save_json=True,          
        project="resultsn_5",       
        name="exp_auto_opt_cos_lr_amp" 
    )

    print("Training completed.")
    print(results) 

if __name__ == "__main__":
    main()


