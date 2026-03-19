from typing import Dict, List, Optional


class ScoreFusion:
    def __init__(
        self,
        rule_weight: float = 0.3,
        ml_weight: float = 0.7,
        amplification_factor: float = 1.5
    ):
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        self.amplification_factor = amplification_factor
        self.labels = {
            0: "Non-Radical",
            1: "Mildly Radical",
            2: "Moderately Radical",
            3: "Highly Radical"
        }

    def fuse(
        self,
        rule_result: Dict,
        ml_result: Dict
    ) -> Dict[str, any]:
        rule_score = rule_result.get('risk_score', 0) / 100.0
        
        if rule_result.get('has_high_risk_terms', False):
            rule_score *= self.amplification_factor
        
        ml_scores = ml_result.get('probabilities', {})
        ml_max_label = ml_result.get('label', 'Non-Radical')
        ml_max_score = ml_result.get('confidence', 0.5)
        
        ml_risk_score = self._compute_ml_risk_score(ml_scores)
        
        fused_score = (
            self.rule_weight * rule_score +
            self.ml_weight * ml_risk_score
        )
        
        final_risk_score = min(int(fused_score * 100), 100)
        final_label = self._determine_label(final_risk_score)
        final_confidence = self._compute_confidence(
            rule_result, ml_result, final_risk_score
        )

        return {
            'label': final_label,
            'risk_score': final_risk_score,
            'confidence': final_confidence,
            'flagged_terms': rule_result.get('matched_terms', []),
            'rule_amplification': rule_result.get('has_high_risk_terms', False),
            'reasoning': self._generate_reasoning(
                rule_result, ml_result, final_risk_score
            )
        }

    def _compute_ml_risk_score(self, ml_scores: Dict[str, float]) -> float:
        weights = {
            'Non-Radical': 0.0,
            'Mildly Radical': 0.33,
            'Moderately Radical': 0.66,
            'Highly Radical': 1.0
        }
        return sum(ml_scores.get(label, 0) * weight 
                   for label, weight in weights.items())

    def _determine_label(self, risk_score: int) -> str:
        if risk_score < 25:
            return "Non-Radical"
        elif risk_score < 50:
            return "Mildly Radical"
        elif risk_score < 75:
            return "Moderately Radical"
        else:
            return "Highly Radical"

    def _compute_confidence(
        self,
        rule_result: Dict,
        ml_result: Dict,
        final_score: int
    ) -> float:
        ml_confidence = ml_result.get('confidence', 0.5)
        rule_certainty = 1.0 if rule_result.get('flagged') else 0.0
        
        base_confidence = (
            self.ml_weight * ml_confidence +
            self.rule_weight * rule_certainty
        )
        
        if rule_result.get('has_high_risk_terms'):
            base_confidence = min(base_confidence * 1.2, 1.0)
        
        return round(base_confidence, 2)

    def _generate_reasoning(
        self,
        rule_result: Dict,
        ml_result: Dict,
        final_score: int
    ) -> str:
        reasons = []
        
        matched_terms = rule_result.get('matched_terms', [])
        if matched_terms:
            high_risk = rule_result.get('keyword_details', {}).get('high_risk', [])
            if high_risk:
                reasons.append(f"High-risk terms detected: {', '.join(high_risk[:3])}")
            else:
                reasons.append(f"Flagged keywords found: {', '.join(matched_terms[:3])}")
        
        ml_label = ml_result.get('label', 'Unknown')
        ml_conf = ml_result.get('confidence', 0)
        reasons.append(f"ML model classified as '{ml_label}' with {ml_conf:.0%} confidence")
        
        if rule_result.get('has_high_risk_terms'):
            reasons.append("Rule engine applied amplification due to high-risk content")
        
        return "; ".join(reasons)
