from langgraph.graph import StateGraph, END
from typing import Dict, List, Any, TypedDict, Optional
import cv2
import os


from agents.imagequality import ImageQualityAgent
from agents.damage_detection import DamageDetectionAgent
from agents.part_identification import PartIdentificationAgent
from agents.severity_assessment import SeverityAssessmentAgent

class ClaimState(TypedDict, total=False):
    claim_id: str
    image_path: str
    image: Optional[Any]
    processed_image: Optional[Any]
    quality_result: Optional[Dict[str, Any]]
    damage_result: Optional[Dict[str, Any]]
    part_result: Optional[Dict[str, Any]]
    severity_result: Optional[Dict[str, Any]]


quality_agent = ImageQualityAgent()
damage_agent = DamageDetectionAgent("C:/Misogi/Vehicle-Damage-Detection/car_damage.pt")  # Use your trained model
part_agent = PartIdentificationAgent("C:/Misogi/Vehicle-Damage-Detection/car_parts.pt")


severity_rules = {
    "minor": {"cost_range": [100, 500], "repair_days": 1},
    "moderate": {"cost_range": [500, 1500], "repair_days": 3},
    "major": {"cost_range": [1500, 5000], "repair_days": 7},
    "severe": {"cost_range": [5000, 15000], "repair_days": 14}
}
severity_agent = SeverityAssessmentAgent(severity_rules)

workflow = StateGraph(ClaimState)

def step_quality(state: ClaimState) -> ClaimState:

    img = cv2.imread(state["image_path"])
    if img is None:
        raise ValueError(f"Could not load image from path: {state['image_path']}")
    
    # Process quality
    result = quality_agent.process(img)
    

    if result["processable"] and "enhanced_image" in result:
        processed_image = result["enhanced_image"]
    else:
        processed_image = img
    
    return {
        **state,
        "image": img,
        "processed_image": processed_image,
        "quality_result": result
    }

workflow.add_node("quality_check", step_quality)

# 2. Damage Detection
def step_damage(state: ClaimState) -> ClaimState:
    img = state["processed_image"]  
    result = damage_agent.process(img)
    
    return {
        "damage_result": result
    }

workflow.add_node("damage_detection", step_damage)

# 3. Part Identification
def step_parts(state: ClaimState) -> ClaimState:
    img = state["processed_image"]
    result = part_agent.process(img)
    
    return {
        "part_result": result
    }

workflow.add_node("part_identification", step_parts)

# 4. Severity Assessment
def step_severity(state: ClaimState) -> ClaimState:
    damage = state["damage_result"]
    parts = state["part_result"]
    result = severity_agent.process(damage, parts)
    
    print(f"Severity Assessment - {result['overall_severity']} (Score: {result['severity_score']})")
    print(f"Estimated Cost: ${result['estimated_cost_range'][0]}-${result['estimated_cost_range'][1]}")
    
    return {
        "severity_result": result
    }

workflow.add_node("severity_assessment", step_severity)



workflow.set_entry_point("quality_check")
workflow.add_edge("quality_check", "damage_detection")
workflow.add_edge("quality_check", "part_identification")
workflow.add_edge("damage_detection", "severity_assessment")
workflow.add_edge("part_identification", "severity_assessment")
workflow.add_edge("severity_assessment", END)


app = workflow.compile()
