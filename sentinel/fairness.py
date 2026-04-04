from typing import Dict, List, Any

class FairnessEvaluator:
    def __init__(self):
        # Sensitive keywords that often lead to bias
        self.sensitive_categories = {
            'religion': ['jihad', 'holy war', 'faith', 'belief', 'righteous'],
            'political': ['struggle', 'resist', 'fight', 'freedom', 'liberation', 'rights', 'justice'],
            'identity': ['our people', 'nation', 'homeland', 'us versus them']
        }
        
    def check_bias_risk(self, text: str, matched_terms: List[str]) -> Dict[str, Any]:
        risk_factors = []
        for category, terms in self.sensitive_categories.items():
            found = [t for t in matched_terms if t.lower() in terms]
            if found:
                risk_factors.append({
                    'category': category,
                    'terms': found
                })
        
        return {
            'high_bias_risk': len(risk_factors) > 0,
            'risk_factors': risk_factors
        }

    def evaluate_fairness_report(self, results: List[Dict]) -> Dict[str, Any]:
        # This would aggregate results over time to see if certain groups are disproportionately flagged
        # For now, it's a placeholder for future extension
        total = len(results)
        flagged = sum(1 for r in results if r.get('label') != 'Non-Radical')
        
        bias_risk_count = sum(1 for r in results if r.get('bias_metadata', {}).get('high_bias_risk', False))
        
        return {
            'total_processed': total,
            'flagged_rate': flagged / max(1, total),
            'bias_risk_rate': bias_risk_count / max(1, total)
        }
