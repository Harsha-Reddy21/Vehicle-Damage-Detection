import cv2
import sys
import os
from pathlib import Path

# Add path to our YOLOv5 inference module
sys.path.append(str(Path(__file__).parent))
from yolov5_inference import YOLOv5Detector

class DamageDetectionAgent:
    def __init__(self, model_path="C:/Misogi/Vehicle-Damage-Detection/best.pt"):
        """
        Initialize damage detection agent with YOLOv5 model.
        
        Args:
            model_path (str): Path to the trained YOLOv5 model weights
        """
        print(f"Initializing DamageDetectionAgent with model: {model_path}")
        self.detector = YOLOv5Detector(model_path)

    def process(self, image):
        """
        Process image to detect vehicle damage.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            dict: Detection results with bounding boxes and total damage area
        """
        # Save image temporarily for YOLOv5 inference
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, image)
        
        try:
            # Run inference
            results = self.detector.detect(temp_path, conf_thres=0.25, save_results=False)
            
            # Process results
            detections = []
            total_area = 0
            img_h, img_w = image.shape[:2]
            
            for result in results:
                for detection in result['detections']:
                    bbox = detection['bbox']  # [x1, y1, x2, y2]
                    conf = detection['confidence']
                    damage_type = detection['class']
                    
                    # Calculate damage area percentage
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    damage_pct = (bbox_area / (img_w * img_h)) * 100
                    total_area += damage_pct
                    
                    detections.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "damage_type": damage_type
                    })
            
            return {
                "detections": detections,
                "total_damage_area": round(total_area, 2)
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    image_path = "C:/Misogi/vehicle_dataset/C9LJUWLH/car-images/front_left-15.jpeg"
    image = cv2.imread(image_path)
    agent = DamageDetectionAgent()
    result = agent.process(image)
    print(result)


