import time
import cv2
from agents.imagequality import ImageQualityAgent
from agents.damage_detection import DamageDetectionAgent
from agents.part_identification import PartIdentificationAgent
from agents.severity_assessment import SeverityAssessmentAgent
from severity_assessment import severity_rules

class DamageAssessmentOrchestrator:
    def __init__(self):
        self.quality_agent = ImageQualityAgent()
        self.damage_agent = DamageDetectionAgent()
        self.part_agent = PartIdentificationAgent()
        self.severity_agent = SeverityAssessmentAgent(severity_rules)

    def process_claim(self, claim):
        start_time = time.time()
        results = {"claim_id": claim["claim_id"], "images": [], "errors": []}
        all_detections, all_parts, severities = [], [], []

        for img_data in claim["images"]:
            try:
                img = cv2.imread(img_data["image_path"])

                
                quality = self.quality_agent.process(img)
                if not quality["processable"]:
                    results["errors"].append(f"Image {img_data['image_id']} not processable")
                    continue

                
                damage = self.damage_agent.process(quality["enhanced_image"])

                
                parts = self.part_agent.process(quality["enhanced_image"], damage["detections"])

                
                severity = self.severity_agent.process(damage, parts)

                
                all_detections.extend(damage["detections"])
                all_parts.extend(parts["damaged_parts"])
                severities.append(severity)

                results["images"].append({
                    "image_id": img_data["image_id"],
                    "quality": quality,
                    "damage": damage,
                    "parts": parts,
                    "severity": severity
                })

            except Exception as e:
                results["errors"].append(f"Processing failed for {img_data['image_id']}: {str(e)}")
                continue

        
        merged_summary = self._merge_results(all_detections, all_parts, severities)

        results["assessment_result"] = merged_summary
        results["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return results

    def _merge_results(self, detections, parts, severities):

        affected_parts = list(set([p["part_name"] for p in parts]))
        overall_severity = max(severities, key=lambda s: s["severity_score"]) if severities else None

        return {
            "damage_summary": {
                "total_damages_found": len(detections),
                "affected_parts": affected_parts,
                "overall_severity": overall_severity["overall_severity"] if overall_severity else "unknown",
                "confidence_score": sum(d["confidence"] for d in detections) / len(detections) if detections else 0
            }
        }



if __name__ == "__main__":
    claim = {
        "claim_id": "123",
        "images": [
            {"image_id": "1", "image_path": "C:/Misogi/vehicle_dataset/C9LJUWLH/car-images/front_left-15.jpeg"}
        ]
    }
    orchestrator = DamageAssessmentOrchestrator()
    orchestrator.process_claim(claim)