# yolov5_inference.py

import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
import time

# Add YOLOv5 path to sys.path
yolov5_path = Path(__file__).parent.parent / 'yolov5'
sys.path.append(str(yolov5_path))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device

class YOLOv5Detector:
    """
    YOLOv5 Detector class that loads the model once and reuses it for multiple inferences.
    """
    
    def __init__(self, weights_path, device='', img_size=640):
        """
        Initialize the YOLOv5 detector.
        
        Args:
            weights_path (str): Path to your trained YOLOv5 weights (e.g., best.pt).
            device (str): Device to run inference on ('', 'cpu', '0', '1', etc.).
            img_size (int): Image size for inference (default=640).
        """
        print("Loading YOLOv5 model...")
        start_time = time.time()
        
        # Initialize
        self.device = select_device(device)
        self.img_size = img_size
        
        # Load model
        self.model = attempt_load(weights_path, device=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # Warmup (skip for segmentation models)
        if hasattr(self.model, 'warmup'):
            self.model.warmup(imgsz=(1, 3, img_size, img_size))
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f}s")
        print(f"Classes: {list(self.names.values())}")
    
    def detect(self, source, conf_thres=0.25, save_results=True, save_dir='runs/detect/exp', 
               save_original=True, save_annotated=True):
        """
        Run inference on a single image or batch of images.
        
        Args:
            source (str): Path to image/video, folder, URL, or 0 for webcam.
            conf_thres (float): Confidence threshold (default=0.25).
            save_results (bool): Whether to save results (controls both original and annotated).
            save_dir (str): Directory to save results.
            save_original (bool): Whether to save original images.
            save_annotated (bool): Whether to save annotated images.
            
        Returns:
            list: List of detection results for each image.
        """
        start_time = time.time()
        
        # Dataloader
        dataset = LoadImages(source, img_size=self.img_size, stride=self.stride, auto=True)
        
        results = []
        
        for path, im, im0s, vid_cap, s in dataset:
            # Prepare image
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if hasattr(self.model, 'fp16') and self.model.fp16 else im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

            # Process predictions
            for i, det in enumerate(pred):
                p, im0 = path, im0s.copy()
                
                detections = []
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Collect results
                    for *xyxy, conf, cls in reversed(det):
                        class_name = self.names[int(cls)]
                        confidence = float(conf)
                        bbox = [int(x) for x in xyxy]  # [x1, y1, x2, y2]
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        }
                        detections.append(detection)

                # Handle image saving
                if save_results and (save_original or save_annotated):
                    # Create save directories
                    base_save_dir = Path(save_dir)
                    original_dir = base_save_dir / 'original'
                    annotated_dir = base_save_dir / 'annotated'
                    
                    # Get filename for saving
                    image_name = Path(p).name
                    
                    # Save original image
                    if save_original:
                        original_dir.mkdir(parents=True, exist_ok=True)
                        original_path = original_dir / image_name
                        cv2.imwrite(str(original_path), im0s.copy())
                    
                    # Save annotated image (only if there are detections or user wants all images)
                    if save_annotated:
                        annotated_dir.mkdir(parents=True, exist_ok=True)
                        annotated_path = annotated_dir / image_name
                        
                        if len(det):
                            # Annotate image with detections
                            annotator = Annotator(im0, line_width=3, example=str(self.names))
                            for *xyxy, conf, cls in reversed(det):
                                class_name = self.names[int(cls)]
                                label = f'{class_name} {conf:.2f}'
                                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                            im0_annotated = annotator.result()
                        else:
                            # Save copy of original if no detections
                            im0_annotated = im0.copy()
                        
                        cv2.imwrite(str(annotated_path), im0_annotated)
                
                # Prepare result with save paths
                result = {
                    'image_path': p,
                    'detections': detections,
                    'detection_count': len(detections),
                    'saved_paths': {}
                }
                
                # Add saved paths to result if images were saved
                if save_results and (save_original or save_annotated):
                    base_save_dir = Path(save_dir)
                    image_name = Path(p).name
                    
                    if save_original:
                        result['saved_paths']['original'] = str(base_save_dir / 'original' / image_name)
                    if save_annotated:
                        result['saved_paths']['annotated'] = str(base_save_dir / 'annotated' / image_name)
                
                results.append(result)
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.3f}s")
        
        return results
    
    def print_results(self, results):
        """Print detection results in a formatted way."""
        for result in results:
            print(f"\nDetection Results for {result['image_path']}:")
            if result['detection_count'] > 0:
                for detection in result['detections']:
                    print(f"  {detection['class']}: {detection['confidence']:.2f} "
                          f"[{', '.join(map(str, detection['bbox']))}]")
                print(f"  Total detections: {result['detection_count']}")
            else:
                print("  No detections found")
            
            # Print saved paths if available
            if 'saved_paths' in result and result['saved_paths']:
                print("  Saved images:")
                for save_type, path in result['saved_paths'].items():
                    print(f"    {save_type.capitalize()}: {path}")

def run_inference(weights_path, source, img_size=640, conf_thres=0.25, save_results=True, 
                  save_dir='runs/detect/exp', save_original=True, save_annotated=True):
    """
    Legacy function for backward compatibility.
    Creates a detector instance and runs inference.
    
    Args:
        weights_path (str): Path to YOLOv5 weights file.
        source (str): Path to image/video, folder, URL, or 0 for webcam.
        img_size (int): Image size for inference.
        conf_thres (float): Confidence threshold.
        save_results (bool): Whether to save results.
        save_dir (str): Directory to save results.
        save_original (bool): Whether to save original images.
        save_annotated (bool): Whether to save annotated images.
    """
    detector = YOLOv5Detector(weights_path, img_size=img_size)
    results = detector.detect(source, conf_thres=conf_thres, save_results=save_results,
                             save_dir=save_dir, save_original=save_original, 
                             save_annotated=save_annotated)
    detector.print_results(results)
    return results

if __name__ == "__main__":
    # Example usage
    weights = "C:/Misogi/Vehicle-Damage-Detection/best.pt"  # your trained weights path
    
    # Method 1: Using the class for efficient multiple inferences
    print("=== Method 1: Using YOLOv5Detector class (recommended for multiple images) ===")
    detector = YOLOv5Detector(weights)
    
    # Run inference on multiple images efficiently (model loaded only once)
    test_images = [
        "C:/Misogi/vehicle_dataset/C4VE7MOV/car-images/front_left-15.jpeg",
        # Add more image paths here for batch processing
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            # Example with enhanced storage options
            results = detector.detect(
                image_path, 
                conf_thres=0.25,
                save_results=True,
                save_dir='runs/detect/enhanced_storage',
                save_original=True,  # Save original images
                save_annotated=True  # Save annotated images
            )
            detector.print_results(results)
        else:
            print(f"Image not found: {image_path}")
    
    print("\n" + "="*60)
    print("=== Method 2: Using legacy function (for single inference) ===")
    # Method 2: Legacy function (loads model each time - slower for multiple images)
    single_image = "C:/Misogi/vehicle_dataset/C4VE7MOV/car-images/front_left-15.jpeg"
    if os.path.exists(single_image):
        run_inference(
            weights, 
            single_image,
            save_results=True,
            save_dir='runs/detect/legacy_storage',
            save_original=True,
            save_annotated=True
        )
