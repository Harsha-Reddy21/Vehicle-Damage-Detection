import random
import cv2
    
CAR_PARTS = {
    "front_bumper": "FB-001",
    "hood": "HD-002",
    "left_headlight": "LH-003",
    "right_headlight": "RH-004",
    "rear_bumper": "RB-005"
}

class PartIdentificationAgent:
    def process(self, image, damage_bboxes):
        results = []
        for det in damage_bboxes:
            part_name, part_id = random.choice(list(CAR_PARTS.items()))
            results.append({
                "part_name": part_name,
                "part_id": part_id,
                "damage_percentage": round(random.uniform(10, 50), 2),
                "bbox": det["bbox"]
            })
        return {"damaged_parts": results}




