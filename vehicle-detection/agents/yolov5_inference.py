# yolov5_inference.py

import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np

# Add YOLOv5 path to sys.path
yolov5_path = Path(__file__).parent.parent / 'yolov5'
sys.path.append(str(yolov5_path))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device

def run_inference(weights_path, source, img_size=640, conf_thres=0.25):
    """
    Run YOLOv5 inference on an image/video/webcam source.

    Args:
        weights_path (str): Path to your trained YOLOv5 weights (e.g., best.pt).
        source (str/int): Path to image/video, folder, URL, or 0 for webcam.
        img_size (int): Image size for inference (default=640).
        conf_thres (float): Confidence threshold (default=0.25).
    """

    # Initialize
    device = select_device('')
    
    # Load model
    model = attempt_load(weights_path, device=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names

    # Dataloader
    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=True)

    # Run inference
    # Warmup (skip for segmentation models)
    if hasattr(model, 'warmup'):
        model.warmup(imgsz=(1, 3, img_size, img_size))
    
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if hasattr(model, 'fp16') and model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0 = path, im0s.copy()
            
            print(f"\nDetection Results for {p}:")
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for *xyxy, conf, cls in reversed(det):
                    class_name = names[int(cls)]
                    confidence = float(conf)
                    print(f"  {class_name}: {confidence:.2f}")

                # Annotate image
                annotator = Annotator(im0, line_width=3, example=str(names))
                for *xyxy, conf, cls in reversed(det):
                    class_name = names[int(cls)]
                    label = f'{class_name} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                # Save/show results
                im0 = annotator.result()
                
                # Save image
                save_path = str(Path('runs/detect/exp') / Path(p).name)
                os.makedirs(Path(save_path).parent, exist_ok=True)
                cv2.imwrite(save_path, im0)
                print(f"  Results saved to {save_path}")
                
                # Show image (optional - comment out if running headless)
                # cv2.imshow('YOLOv5 Detection', im0)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                print("  No detections found")

    return True

if __name__ == "__main__":
    # Example usage
    weights = "C:/Misogi/Vehicle-Damage-Detection/best.pt"  # your trained weights path
    source = "C:/Misogi/vehicle_dataset/C4VE7MOV/car-images/front_left-15.jpeg"  # image/video path or 0 for webcam

    run_inference(weights, source)
