import numpy as np
from typing import Dict, List
import re


class InterpretabilityMetrics:
    def __init__(self, config):
        self.config = config
    
    def evaluate_clarity(self, explanation: str) -> float:
        
        words = explanation.split()
        num_words = len(words)
        
        if num_words < 10:
            length_score = 0.5
        elif num_words <= 100:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (num_words - 100) / 200)
        
        sentences = explanation.split('.')
        num_sentences = len([s for s in sentences if len(s.strip()) > 0])
        
        if num_sentences == 0:
            structure_score = 0.0
        else:
            avg_sentence_length = num_words / num_sentences
            if 10 <= avg_sentence_length <= 25:
                structure_score = 1.0
            else:
                structure_score = max(0.3, 1.0 - abs(avg_sentence_length - 17.5) / 20)
        
        technical_terms = ['severity', 'stress', 'incident', 'operator', 'system', 
                          'attack', 'anomaly', 'cognitive', 'load', 'response']
        
        term_count = sum(1 for term in technical_terms if term.lower() in explanation.lower())
        terminology_score = min(1.0, term_count / 5)
        
        clarity_score = 0.4 * length_score + 0.3 * structure_score + 0.3 * terminology_score
        
        return float(clarity_score)
    
    def evaluate_relevance(
        self,
        explanation: str,
        severity: int,
        stress: int
    ) -> float:
        
        severity_terms = {
            0: ['normal', 'baseline', 'stable', 'routine'],
            1: ['low', 'minor', 'slight'],
            2: ['medium', 'moderate', 'elevated'],
            3: ['high', 'severe', 'critical', 'urgent']
        }
        
        stress_terms = {
            0: ['baseline', 'calm', 'relaxed', 'normal'],
            1: ['low', 'slight', 'mild'],
            2: ['medium', 'moderate', 'elevated'],
            3: ['high', 'severe', 'intense', 'extreme']
        }
        
        explanation_lower = explanation.lower()
        
        severity_match = any(term in explanation_lower for term in severity_terms.get(severity, []))
        stress_match = any(term in explanation_lower for term in stress_terms.get(stress, []))
        
        context_terms = ['incident', 'attack', 'operator', 'response', 'system']
        context_count = sum(1 for term in context_terms if term in explanation_lower)
        context_score = min(1.0, context_count / 3)
        
        relevance_score = (
            0.4 * (1.0 if severity_match else 0.3) +
            0.4 * (1.0 if stress_match else 0.3) +
            0.2 * context_score
        )
        
        return float(relevance_score)
    
    def evaluate_usefulness(
        self,
        explanation: str,
        explanation_type: str = 'rationale'
    ) -> float:
        
        if explanation_type == 'rationale':
            required_elements = ['why', 'because', 'due to', 'caused by', 'result']
        elif explanation_type == 'counterfactual':
            required_elements = ['if', 'would', 'could', 'change', 'different']
        else:
            required_elements = ['cause', 'effect', 'relationship', 'influence']
        
        explanation_lower = explanation.lower()
        element_count = sum(1 for elem in required_elements if elem in explanation_lower)
        element_score = min(1.0, element_count / 2)
        
        actionable_terms = ['should', 'recommend', 'suggest', 'need', 'must', 'consider']
        actionable_count = sum(1 for term in actionable_terms if term in explanation_lower)
        actionable_score = min(1.0, actionable_count / 2)
        
        usefulness_score = 0.6 * element_score + 0.4 * actionable_score
        
        return float(usefulness_score)
    
    def compute_cf_consistency(
        self,
        counterfactuals: List[str],
        severity_pairs: List[Tuple[int, int]],
        stress_pairs: List[Tuple[int, int]]
    ) -> float:
        
        consistency_scores = []
        
        for i, cf in enumerate(counterfactuals):
            sev_curr, sev_new = severity_pairs[i]
            stress_curr, stress_new = stress_pairs[i]
            
            cf_lower = cf.lower()
            
            severity_change_mentioned = any(term in cf_lower for term in ['severity', 'incident', 'attack'])
            stress_change_mentioned = any(term in cf_lower for term in ['stress', 'cognitive', 'operator'])
            
            directional_terms = ['increase', 'decrease', 'higher', 'lower', 'more', 'less']
            direction_mentioned = any(term in cf_lower for term in directional_terms)
            
            consistency = (
                0.4 * (1.0 if severity_change_mentioned else 0.0) +
                0.4 * (1.0 if stress_change_mentioned else 0.0) +
                0.2 * (1.0 if direction_mentioned else 0.0)
            )
            
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return float(avg_consistency)
    
    def evaluate_explanation_quality(
        self,
        explanations: Dict[str, List[str]],
        severities: np.ndarray,
        stresses: np.ndarray
    ) -> Dict[str, float]:
        
        rationales = explanations.get('rationales', [])
        counterfactuals = explanations.get('counterfactuals', [])
        
        clarity_scores = []
        relevance_scores = []
        usefulness_scores = []
        
        for i, rationale in enumerate(rationales):
            if i < len(severities):
                clarity = self.evaluate_clarity(rationale)
                relevance = self.evaluate_relevance(rationale, int(severities[i]), int(stresses[i]))
                usefulness = self.evaluate_usefulness(rationale, 'rationale')
                
                clarity_scores.append(clarity)
                relevance_scores.append(relevance)
                usefulness_scores.append(usefulness)
        
        cf_consistency = 0.0
        if counterfactuals:
            severity_pairs = [(int(severities[i]), max(0, int(severities[i])-1)) 
                            for i in range(min(len(counterfactuals), len(severities)))]
            stress_pairs = [(int(stresses[i]), max(0, int(stresses[i])-1)) 
                          for i in range(min(len(counterfactuals), len(stresses)))]
            cf_consistency = self.compute_cf_consistency(counterfactuals, severity_pairs, stress_pairs)
        
        return {
            'avg_clarity': float(np.mean(clarity_scores)) if clarity_scores else 0.0,
            'avg_relevance': float(np.mean(relevance_scores)) if relevance_scores else 0.0,
            'avg_usefulness': float(np.mean(usefulness_scores)) if usefulness_scores else 0.0,
            'cf_consistency': cf_consistency,
            'overall_quality': float(np.mean([
                np.mean(clarity_scores) if clarity_scores else 0.0,
                np.mean(relevance_scores) if relevance_scores else 0.0,
                np.mean(usefulness_scores) if usefulness_scores else 0.0,
                cf_consistency
            ]))
        }