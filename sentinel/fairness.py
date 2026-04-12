from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics


class FairnessMetrics:
    DISPARATE_IMPACT_THRESHOLD = 0.8
    EQUALIZED_ODDS_BUFFER = 0.1
    
    @staticmethod
    def disparate_impact_ratio(group_positive_rate: float, reference_positive_rate: float) -> float:
        if reference_positive_rate == 0:
            return 0.0
        return group_positive_rate / reference_positive_rate
    
    @staticmethod
    def demographic_parity_difference(group_positive_rate: float, reference_positive_rate: float) -> float:
        return abs(group_positive_rate - reference_positive_rate)
    
    @staticmethod
    def equalized_odds_difference(group_tpr: float, reference_tpr: float, group_fpr: float, reference_fpr: float) -> float:
        tpr_diff = abs(group_tpr - reference_tpr)
        fpr_diff = abs(group_fpr - reference_fpr)
        return (tpr_diff + fpr_diff) / 2


class FairnessEvaluator:
    def __init__(self):
        self.sensitive_categories = {
            'religion': ['jihad', 'holy war', 'faith', 'belief', 'righteous'],
            'political': ['struggle', 'resist', 'fight', 'freedom', 'liberation', 'rights', 'justice'],
            'identity': ['our people', 'nation', 'homeland', 'us versus them']
        }
        self._result_history: List[Dict] = []
        self._category_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'positive': 0, 'total': 0})
    
    def check_bias_risk(self, text: str, matched_terms: List[str]) -> Dict[str, Any]:
        risk_factors = []
        detected_categories = set()
        
        for category, terms in self.sensitive_categories.items():
            found = [t for t in matched_terms if t.lower() in terms]
            if found:
                risk_factors.append({
                    'category': category,
                    'terms': found
                })
                detected_categories.add(category)
        
        return {
            'high_bias_risk': len(risk_factors) > 0,
            'risk_factors': risk_factors,
            'detected_categories': list(detected_categories)
        }
    
    def record_result(self, text: str, label: str, label_id: int, matched_terms: List[str]) -> None:
        result = {
            'text': text,
            'label': label,
            'label_id': label_id,
            'matched_terms': matched_terms,
            'is_radical': label != 'Non-Radical'
        }
        self._result_history.append(result)
        
        bias_check = self.check_bias_risk(text, matched_terms)
        for category in bias_check.get('detected_categories', []):
            self._category_stats[category]['total'] += 1
            if result['is_radical']:
                self._category_stats[category]['positive'] += 1
    
    def evaluate_fairness_report(self, results: List[Dict], sensitive_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return self._empty_fairness_report()
        
        flagged = [r for r in results if r.get('label') != 'Non-Radical']
        flagged_count = len(flagged)
        
        overall_positive_rate = flagged_count / total
        
        category_metrics = self._compute_category_metrics(results, sensitive_groups)
        
        reference_rate = overall_positive_rate
        disparate_impact_concerns = []
        
        for category, stats in category_metrics.items():
            if stats['total'] > 0:
                category_positive_rate = stats['positive'] / stats['total']
                di_ratio = FairnessMetrics.disparate_impact_ratio(category_positive_rate, reference_rate)
                
                if di_ratio < FairnessMetrics.DISPARATE_IMPACT_THRESHOLD or di_ratio > 1 / FairnessMetrics.DISPARATE_IMPACT_THRESHOLD:
                    disparate_impact_concerns.append({
                        'category': category,
                        'positive_rate': round(category_positive_rate, 4),
                        'reference_rate': round(reference_rate, 4),
                        'disparate_impact_ratio': round(di_ratio, 4),
                        'severity': 'high' if di_ratio < 0.5 else 'medium'
                    })
        
        bias_risk_count = sum(1 for r in results if r.get('bias_metadata', {}).get('high_bias_risk', False))
        
        return {
            'total_processed': total,
            'flagged_count': flagged_count,
            'flagged_rate': round(flagged_count / total, 4),
            'bias_risk_count': bias_risk_count,
            'bias_risk_rate': round(bias_risk_count / total, 4),
            'category_metrics': category_metrics,
            'disparate_impact_concerns': sorted(
                disparate_impact_concerns, 
                key=lambda x: x['severity'] == 'high', 
                reverse=True
            ),
            'fairness_score': self._compute_fairness_score(disparate_impact_concerns, total),
            'requires_fairness_review': len([c for c in disparate_impact_concerns if c['severity'] == 'high']) > 0
        }
    
    def _compute_category_metrics(self, results: List[Dict], sensitive_groups: Optional[Dict[str, List[str]]]) -> Dict[str, Dict[str, int]]:
        category_stats = defaultdict(lambda: {'positive': 0, 'total': 0})
        
        for result in results:
            matched_terms = result.get('flagged_terms', [])
            bias_check = self.check_bias_risk(result.get('text', ''), matched_terms)
            
            for category in bias_check.get('detected_categories', []):
                category_stats[category]['total'] += 1
                if result.get('label') != 'Non-Radical':
                    category_stats[category]['positive'] += 1
        
        return dict(category_stats)
    
    def _compute_fairness_score(self, concerns: List[Dict], total: int) -> float:
        if total == 0:
            return 1.0
        
        high_severity = sum(1 for c in concerns if c['severity'] == 'high')
        medium_severity = sum(1 for c in concerns if c['severity'] == 'medium')
        
        penalty = (high_severity * 0.3 + medium_severity * 0.1) / max(1, len(concerns) if concerns else 1)
        
        score = max(0.0, 1.0 - penalty)
        return round(score, 4)
    
    def _empty_fairness_report(self) -> Dict[str, Any]:
        return {
            'total_processed': 0,
            'flagged_count': 0,
            'flagged_rate': 0.0,
            'bias_risk_count': 0,
            'bias_risk_rate': 0.0,
            'category_metrics': {},
            'disparate_impact_concerns': [],
            'fairness_score': 1.0,
            'requires_fairness_review': False
        }
    
    def get_historical_stats(self) -> Dict[str, Any]:
        if not self._result_history:
            return {'total': 0, 'message': 'No historical data available'}
        
        return {
            'total_results': len(self._result_history),
            'category_breakdown': dict(self._category_stats)
        }
    
    def reset_history(self) -> None:
        self._result_history.clear()
        self._category_stats.clear()
