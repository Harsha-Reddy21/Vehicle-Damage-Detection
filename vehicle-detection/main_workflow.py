from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
import random


# -----------------------------
class ClaimState(Dict[str, Any]):
    pass


class ImageQualityAgent:
    def process(self, image: str) -> Dict[str, Any]:
        return {
            "quality_score": round(random.uniform(0.7, 0.95), 2),
            "issues": ["blur"] if random.random() < 0.3 else [],
            "processable": True,
            "enhanced_image": image + "_enhanced"
        }


class DamageDetectionAgent:
    def process(self, image: str) -> Dict[str, Any]:
        return {
            "detections": [
                {
                    "bbox": [120, 340, 450, 520],
                    "confidence": 0.92,
                    "damage_type": random.choice(["dent", "scratch", "crack"])
                }
            ],
            "total_damage_area": round(random.uniform(10, 25), 2)
        }


class PartIdentificationAgent:
    def process(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "damaged_parts": [
                {
                    "part_name": "front_bumper",
                    "part_id": "FB-001",
                    "damage_percentage": 30,
                    "bbox": detections[0]["bbox"]
                }
            ]
        }


class SeverityAssessmentAgent:
    def process(self, damage_data, part_data) -> Dict[str, Any]:
        return {
            "overall_severity": "moderate",
            "severity_score": 6.5,
            "repair_category": "body_shop_required",
            "estimated_cost_range": [2000, 3500],
            "repair_time_days": 3
        }


# -----------------------------
# Orchestrator using LangGraph
# -----------------------------

quality_agent = ImageQualityAgent()
damage_agent = DamageDetectionAgent()
part_agent = PartIdentificationAgent()
severity_agent = SeverityAssessmentAgent()

# Define workflow graph
workflow = StateGraph(ClaimState)

# 1. Image Quality
def step_quality(state: ClaimState):
    img = state["image"]
    result = quality_agent.process(img)
    state["quality_result"] = result
    return state

workflow.add_node("quality_check", step_quality)

# 2. Damage Detection
def step_damage(state: ClaimState):
    img = state["image"]
    result = damage_agent.process(img)
    state["damage_result"] = result
    return state

workflow.add_node("damage_detection", step_damage)

# 3. Part Identification
def step_parts(state: ClaimState):
    detections = state["damage_result"]["detections"]
    result = part_agent.process(detections)
    state["part_result"] = result
    return state

workflow.add_node("part_identification", step_parts)

# 4. Severity Assessment
def step_severity(state: ClaimState):
    damage = state["damage_result"]
    parts = state["part_result"]
    result = severity_agent.process(damage, parts)
    state["severity_result"] = result
    return state

workflow.add_node("severity_assessment", step_severity)

# -----------------------------
# Define Edges (flow)
# -----------------------------

workflow.set_entry_point("quality_check")
workflow.add_edge("quality_check", "damage_detection")
workflow.add_edge("damage_detection", "part_identification")
workflow.add_edge("part_identification", "severity_assessment")
workflow.add_edge("severity_assessment", END)

# Compile graph
app = workflow.compile()

# -----------------------------
# Run Example
# -----------------------------
if __name__ == "__main__":
    input_state = ClaimState({
        "claim_id": "CLM-2024-001",
        "image": "IMG-001.jpg"
    })
    final_state = app.invoke(input_state)
    print("\n--- Final Assessment ---")
    print(final_state)
