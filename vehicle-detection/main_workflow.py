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

# Initialize agents
quality_agent = ImageQualityAgent()
damage_agent = DamageDetectionAgent("C:/Misogi/Vehicle-Damage-Detection/best.pt")  # Use your trained model
part_agent = PartIdentificationAgent()


severity_rules = {
    "minor": {"cost_range": [100, 500], "repair_days": 1},
    "moderate": {"cost_range": [500, 1500], "repair_days": 3},
    "major": {"cost_range": [1500, 5000], "repair_days": 7},
    "severe": {"cost_range": [5000, 15000], "repair_days": 14}
}
severity_agent = SeverityAssessmentAgent(severity_rules)

workflow = StateGraph(ClaimState)

# 1. Image Quality Check
def step_quality(state: ClaimState) -> ClaimState:
    print(f"Quality Check - Processing image: {state['image_path']}")
    
    # Load image
    img = cv2.imread(state["image_path"])
    if img is None:
        raise ValueError(f"Could not load image from path: {state['image_path']}")
    
    # Process quality
    result = quality_agent.process(img)
    
    print(f"Quality Check - Score: {result['quality_score']}, Issues: {result['issues']}")
    
    # Use enhanced image for further processing if available
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
    img = state["processed_image"]  # Use processed/enhanced image
    result = damage_agent.process(img)
    
    print(f"Damage Detection - Found {len(result['detections'])} damages, Total area: {result['total_damage_area']}%")
    
    return {
        **state,
        "damage_result": result
    }

workflow.add_node("damage_detection", step_damage)

# 3. Part Identification
def step_parts(state: ClaimState) -> ClaimState:
    img = state["processed_image"]
    detections = state["damage_result"]["detections"]
    result = part_agent.process(img, detections)
    
    print(f"Part Identification - {len(result['damaged_parts'])} parts identified")
    
    return {
        **state,
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
        **state,
        "severity_result": result
    }

workflow.add_node("severity_assessment", step_severity)


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
    # Example with actual image path
    image_path = "C:/Misogi/vehicle_dataset/C4VE7MOV/car-images/front_left-15.jpeg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid image path")
        exit(1)
    
    print("=== Vehicle Damage Assessment Workflow ===")
    print(f"Processing image: {image_path}")
    print("-" * 50)
    
    input_state = ClaimState({
        "claim_id": "CLM-2024-001",
        "image_path": image_path
    })
    
    try:
        final_state = app.invoke(input_state)
        
        print("\n" + "=" * 50)
        print("--- FINAL ASSESSMENT REPORT ---")
        print("=" * 50)
        
        # Quality Summary
        quality = final_state.get("quality_result", {})
        print(f"Image Quality Score: {quality.get('quality_score', 'N/A')}")
        print(f"Quality Issues: {quality.get('issues', [])}")
        
        # Damage Summary
        damage = final_state.get("damage_result", {})
        print(f"Damages Detected: {len(damage.get('detections', []))}")
        print(f"Total Damage Area: {damage.get('total_damage_area', 0)}%")
        
        # Parts Summary
        parts = final_state.get("part_result", {})
        damaged_parts = parts.get("damaged_parts", [])
        print(f"Damaged Parts: {len(damaged_parts)}")
        for part in damaged_parts:
            print(f"  - {part['part_name']}: {part['damage_percentage']}% damaged")
        
        # Severity Summary
        severity = final_state.get("severity_result", {})
        print(f"Overall Severity: {severity.get('overall_severity', 'N/A')}")
        print(f"Repair Category: {severity.get('repair_category', 'N/A')}")
        cost_range = severity.get('estimated_cost_range', [0, 0])
        print(f"Estimated Repair Cost: ${cost_range[0]} - ${cost_range[1]}")
        print(f"Estimated Repair Time: {severity.get('repair_time_days', 0)} days")
        
    except Exception as e:
        print(f"Error processing workflow: {e}")
        import traceback
        traceback.print_exc()
