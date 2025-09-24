from ultralytics import YOLO
import cv2

class DamageDetectionAgent:
    def __init__(self, model_path="yolov8n.pt"):
        
        self.model = YOLO(model_path)

    def process(self, image):
        
        results = self.model(image)
        detections = []
        total_area = 0
        img_h, img_w = image.shape[:2]

        for r in results[0].boxes:
            x1, y1, x2, y2 = r.xyxy[0].tolist()
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            damage_type = self.model.names[cls]

            
            bbox_area = (x2 - x1) * (y2 - y1)
            damage_pct = (bbox_area / (img_w * img_h)) * 100
            total_area += damage_pct

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "damage_type": damage_type
            })

        return {
            "detections": detections,
            "total_damage_area": round(total_area, 2)
        }


if __name__ == "__main__":
    image_path = "C:/Misogi/vehicle_dataset/C9LJUWLH/car-images/front_left-15.jpeg"
    image = cv2.imread(image_path)
    agent = DamageDetectionAgent()
    result = agent.process(image)
    print(result)


