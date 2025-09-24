class SeverityAssessmentAgent:
    def __init__(self, severity_rules):
        self.severity_rules = severity_rules 

    def process(self, damage_data, part_data):
        
        num_parts = len(part_data["damaged_parts"])
        total_area = damage_data["total_damage_area"]

        
        score = 0
        score += num_parts * 2
        score += total_area / 10  # scale area contribution

        
        critical_parts = {"windshield", "engine", "airbags", "hood"}
        for part in part_data["damaged_parts"]:
            if part["part_name"] in critical_parts:
                score += 3

        
        if score < 3:
            severity = "minor"
        elif score < 7:
            severity = "moderate"
        elif score < 12:
            severity = "major"
        else:
            severity = "severe"

        
        cost_range = self.severity_rules[severity]["cost_range"]
        repair_days = self.severity_rules[severity]["repair_days"]

        
        return {
            "overall_severity": severity,
            "severity_score": round(score, 2),
            "repair_category": "body_shop_required" if severity != "minor" else "cosmetic_repair",
            "estimated_cost_range": cost_range,
            "repair_time_days": repair_days
        }


severity_rules = {
    "minor": {"cost_range": [100, 200], "repair_days": 1},
    "moderate": {"cost_range": [200, 400], "repair_days": 2},
    "major": {"cost_range": [400, 800], "repair_days": 3},
    "severe": {"cost_range": [800, 1600], "repair_days": 4}
}

if __name__ == "__main__":
    damage_data = {
    "total_damage_area": 15.5,
    "detections": [
        {"bbox": [100, 200, 400, 500], "confidence": 0.92, "damage_type": "dent"}
    ]
    }

    part_data = {
        "damaged_parts": [
            {"part_name": "front_bumper", "damage_percentage": 30, "bbox": [100, 200, 400, 500]},
            {"part_name": "hood", "damage_percentage": 20, "bbox": [120, 220, 420, 520]}
        ]
    }

    agent = SeverityAssessmentAgent(severity_rules)
    result = agent.process(damage_data, part_data)
    print(result)


