import re
import yaml
from typing import Dict, List, Optional
from pathlib import Path


class RuleEngine:
    def __init__(self, rules_path: str = "data/rules/keywords.yaml"):
        self.rules_path = Path(rules_path)
        self.keywords: Dict[str, List[str]] = {}
        self.patterns: Dict[str, List[str]] = {}
        self.severity_weights: Dict[str, int] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")
        
        with open(self.rules_path, 'r') as f:
            rules = yaml.safe_load(f)
        
        self.keywords = rules.get('keywords', {})
        self.patterns = rules.get('patterns', {})
        self.severity_weights = rules.get('severity_weights', {})

    def check_keywords(self, text: str) -> Dict[str, any]:
        text_lower = text.lower()
        results = {
            'flagged': False,
            'matched_terms': [],
            'risk_score': 0,
            'severity_breakdown': {}
        }

        for severity, keywords in self.keywords.items():
            matches = [kw for kw in keywords if kw.lower() in text_lower]
            if matches:
                results['matched_terms'].extend(matches)
                weight = self.severity_weights.get(severity, 1)
                results['risk_score'] += len(matches) * weight
                results['severity_breakdown'][severity] = matches
                results['flagged'] = True

        return results

    def check_patterns(self, text: str) -> Dict[str, any]:
        results = {
            'flagged': False,
            'matched_patterns': [],
            'pattern_count': 0
        }

        for category, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    results['matched_patterns'].append({
                        'category': category,
                        'pattern': pattern
                    })
                    results['flagged'] = True
                    results['pattern_count'] += 1

        return results

    def analyze(self, text: str) -> Dict[str, any]:
        keyword_results = self.check_keywords(text)
        pattern_results = self.check_patterns(text)

        total_risk = keyword_results['risk_score']
        if pattern_results['flagged']:
            total_risk += pattern_results['pattern_count'] * 2

        all_matches = keyword_results['matched_terms'].copy()
        for match in pattern_results['matched_patterns']:
            all_matches.append(match['pattern'])

        return {
            'flagged': keyword_results['flagged'] or pattern_results['flagged'],
            'matched_terms': all_matches,
            'risk_score': min(total_risk, 100),
            'keyword_details': keyword_results['severity_breakdown'],
            'pattern_details': pattern_results['matched_patterns'],
            'has_high_risk_terms': len(keyword_results['severity_breakdown'].get('high_risk', [])) > 0
        }
