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



if __name__ == "__main__":
    image_path = "C:/Misogi/vehicle_dataset/C9LJUWLH/car-images/front_left-15.jpeg"
    image = cv2.imread(image_path)
    damage_bboxes = [{"bbox": [100, 200, 400, 500]}]
    agent = PartIdentificationAgent()
    result = agent.process(image, damage_bboxes)
    print(result)


