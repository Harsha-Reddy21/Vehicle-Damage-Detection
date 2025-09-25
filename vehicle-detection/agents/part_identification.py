import cv2
import os
from yolov5_inference import YOLOv5Detector

class PartIdentificationAgent:
    def __init__(self, model_path="C:/Misogi/Vehicle-Damage-Detection/best.pt"):
        self.detector = YOLOv5Detector(model_path)
    
    def process(self, image):
        temp_path = "temp_parts.jpg"
        cv2.imwrite(temp_path, image)
        
        try:
            results = self.detector.detect(temp_path, conf_thres=0.25, save_results=False)
            
            parts = []
            for result in results:
                for detection in result['detections']:
                    parts.append({
                        "part_name": detection['class'],
                        "bbox": detection['bbox'],
                        "confidence": detection['confidence']
                    })
            
            return {"identified_parts": parts}
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    image_path = "C:/Misogi/vehicle_dataset/C9LJUWLH/car-images/front_left-15.jpeg"
    image = cv2.imread(image_path)
    agent = PartIdentificationAgent()
    result = agent.process(image)
    print(result)





