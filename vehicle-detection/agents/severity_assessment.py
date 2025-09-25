class SeverityAssessmentAgent:
    def __init__(self, severity_rules):
        self.severity_rules = severity_rules 

    def process(self, damage_data, part_data):
        
        identified_parts = part_data.get("identified_parts", [])
        num_parts = len(identified_parts)
        total_area = damage_data["total_damage_area"]

        
        score = 0
        score += num_parts * 2
        score += total_area / 10  # scale area contribution

        
        
        if score==0:
            return {
                "overall_severity": "minor",
                "severity_score": 0,
                "repair_category": "cosmetic_repair",
                "estimated_cost_range": [0, 0],
                "repair_time_days": 0
            }
        
        if score < 1:
            severity = "minor"
        elif score < 2:
            severity = "moderate"
        elif score <3 :
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


