"""
Hybrid Ranking System
Combines semantic similarity with preference-based scoring
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import re

class HybridRanker:
    """Ranks colleges based on prompt preferences, not just hard filters"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_ready = True
        except:
            self.model = None
            self.embeddings_ready = False
        
        # Pre-compute college text embeddings
        self.college_embeddings = {}
        if self.embeddings_ready:
            self._precompute_college_embeddings()
        
        # Preference weight patterns detected from prompt
        self.PREFERENCE_PATTERNS = {
            'fee_focused': {
                'keywords': ['cheap', 'affordable', 'low fee', 'budget', 'economical', 'inexpensive'],
                'weight_adjustments': {'fee': 0.4, 'rating': 0.3, 'semantic': 0.3}
            },
            'quality_focused': {
                'keywords': ['best', 'top', 'excellent', 'high quality', 'prestigious', 'reputed'],
                'weight_adjustments': {'rating': 0.5, 'semantic': 0.3, 'fee': 0.2}
            },
            'location_focused': {
                'keywords': ['nearby', 'close', 'convenient', 'accessible', 'in kathmandu', 'local'],
                'weight_adjustments': {'location': 0.4, 'semantic': 0.3, 'rating': 0.2, 'fee': 0.1}
            },
            'facilities_focused': {
                'keywords': ['hostel', 'facilities', 'infrastructure', 'accommodation', 'amenities'],
                'weight_adjustments': {'facilities': 0.4, 'semantic': 0.3, 'rating': 0.2, 'fee': 0.1}
            }
        }
    
    def _precompute_college_embeddings(self):
        """Pre-compute embeddings for all colleges"""
        for idx, row in self.df.iterrows():
            # Create rich text description for each college-course combination
            college_text = self._create_college_description(row)
            embedding = self.model.encode([college_text])[0]
            self.college_embeddings[idx] = {
                'embedding': embedding,
                'text': college_text
            }
    
    def _create_college_description(self, row: pd.Series) -> str:
        """Create rich text description for semantic matching"""
        description_parts = []
        
        # Basic info
        description_parts.append(f"{row['CollegeName']}")
        description_parts.append(f"located in {row['Location']}")
        description_parts.append(f"{row['Type'].lower()} college")
        
        # Course info
        description_parts.append(f"offers {row['CourseName']}")
        
        # Financial info
        fee = row.get('Fee', 0)
        if fee > 0:
            if fee < 200000:
                description_parts.append("affordable tuition fees")
            elif fee > 800000:
                description_parts.append("premium education with higher fees")
            else:
                description_parts.append("moderate tuition fees")
        
        # Quality indicators
        rating = row.get('Rating', 0)
        if rating > 4.5:
            description_parts.append("excellent quality education")
        elif rating > 3.5:
            description_parts.append("good quality education")
        
        # Facilities
        if row.get('HostelAvailability', False):
            description_parts.append("hostel accommodation available")
        
        pass_rate = row.get('PassPercentage', 0)
        if pass_rate > 90:
            description_parts.append("high success rate")
        elif pass_rate > 80:
            description_parts.append("good academic performance")
        
        return " ".join(description_parts)
    
    def detect_preferences(self, prompt: str) -> Dict[str, float]:
        """Detect user preferences from prompt and adjust ranking weights"""
        prompt_lower = prompt.lower()
        
        # Default weights
        weights = {
            'semantic': 0.3,
            'rating': 0.25,
            'fee': 0.25,
            'location': 0.1,
            'facilities': 0.1
        }
        
        # Adjust weights based on detected preferences
        for preference_type, config in self.PREFERENCE_PATTERNS.items():
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in prompt_lower)
            
            if keyword_matches > 0:
                # Apply weight adjustments
                adjustment_strength = min(keyword_matches * 0.3, 0.8)
                
                for factor, weight in config['weight_adjustments'].items():
                    if factor in weights:
                        # Blend with existing weight
                        weights[factor] = (weights[factor] + weight * adjustment_strength) / (1 + adjustment_strength)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def rank_colleges(self, prompt: str, intent: str, entities: Dict, k: int = 5) -> List[Dict[str, Any]]:
        """
        Rank colleges based on prompt preferences
        
        Returns list of ranked colleges with scores breakdown
        """
        # Detect preference weights from prompt
        weights = self.detect_preferences(prompt)
        
        results = []
        
        for idx, row in self.df.iterrows():
            scores = self._calculate_scores(prompt, row, idx, intent, entities)
            
            # Weighted final score
            final_score = (
                weights['semantic'] * scores['semantic_score'] +
                weights['rating'] * scores['rating_score'] +
                weights['fee'] * scores['fee_score'] +
                weights['location'] * scores['location_score'] +
                weights['facilities'] * scores['facilities_score']
            )
            
            result = {
                'college_data': row.to_dict(),
                'final_score': final_score,
                'score_breakdown': scores,
                'weights_used': weights,
                'ranking_explanation': self._explain_ranking(scores, weights)
            }
            
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results[:k]
    
    def _calculate_scores(self, prompt: str, row: pd.Series, idx: int, intent: str, entities: Dict) -> Dict[str, float]:
        """Calculate individual scoring components"""
        scores = {}
        
        # 1. Semantic similarity score
        if self.embeddings_ready and idx in self.college_embeddings:
            prompt_embedding = self.model.encode([prompt])[0]
            college_embedding = self.college_embeddings[idx]['embedding']
            
            semantic_similarity = np.dot(prompt_embedding, college_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(college_embedding)
            )
            scores['semantic_score'] = max(0, semantic_similarity)
        else:
            # Fallback keyword matching
            scores['semantic_score'] = self._keyword_similarity(prompt, row)
        
        # 2. Rating score (normalized)
        rating = row.get('Rating', 0)
        scores['rating_score'] = min(rating / 5.0, 1.0)
        
        # 3. Fee score (inverse - lower fee = higher score)
        fee = row.get('Fee', 0)
        if fee > 0:
            # Normalize against reasonable fee ranges
            max_reasonable_fee = 1500000  # 15 lakhs
            fee_ratio = min(fee / max_reasonable_fee, 1.0)
            scores['fee_score'] = 1.0 - fee_ratio
        else:
            scores['fee_score'] = 0.5
        
        # 4. Location score
        scores['location_score'] = self._location_match_score(entities.get('location_mentioned'), row.get('Location', ''))
        
        # 5. Facilities score
        scores['facilities_score'] = self._facilities_score(row)
        
        return scores
    
    def _keyword_similarity(self, prompt: str, row: pd.Series) -> float:
        """Fallback keyword-based similarity when embeddings unavailable"""
        prompt_words = set(prompt.lower().split())
        
        # Create searchable text from row
        searchable_text = f"{row['CollegeName']} {row['Location']} {row['CourseName']} {row['Type']}"
        row_words = set(searchable_text.lower().split())
        
        if len(prompt_words) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(prompt_words & row_words)
        union = len(prompt_words | row_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _location_match_score(self, mentioned_location: str, college_location: str) -> float:
        """Score based on location match"""
        if not mentioned_location or not college_location:
            return 0.5  # Neutral score
        
        mentioned_location = mentioned_location.lower()
        college_location = college_location.lower()
        
        if mentioned_location in college_location:
            return 1.0
        elif any(word in college_location for word in mentioned_location.split()):
            return 0.7
        else:
            return 0.2
    
    def _facilities_score(self, row: pd.Series) -> float:
        """Score based on facilities available"""
        score = 0.0
        
        # Hostel availability
        if row.get('HostelAvailability', False):
            score += 0.4
        
        # Pass percentage as quality indicator
        pass_rate = row.get('PassPercentage', 0)
        if pass_rate > 90:
            score += 0.3
        elif pass_rate > 80:
            score += 0.2
        
        # Faculty ratio (if available)
        faculty_ratio = row.get('FacultyToStudentRatio', 0)
        if faculty_ratio > 0.015:  # Good ratio
            score += 0.3
        
        return min(score, 1.0)
    
    def _explain_ranking(self, scores: Dict[str, float], weights: Dict[str, float]) -> str:
        """Generate human-readable ranking explanation"""
        explanations = []
        
        # Find top contributing factors
        weighted_scores = {
            factor: scores.get(f"{factor}_score", 0) * weights.get(factor, 0)
            for factor in ['semantic', 'rating', 'fee', 'location', 'facilities']
        }
        
        top_factors = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        factor_names = {
            'semantic': 'relevance to query',
            'rating': 'college rating',
            'fee': 'affordability',
            'location': 'location match',
            'facilities': 'facilities quality'
        }
        
        for factor, contribution in top_factors:
            if contribution > 0.1:  # Only mention significant contributors
                explanations.append(f"high {factor_names.get(factor, factor)}")
        
        return f"Ranked highly due to: {', '.join(explanations)}" if explanations else "Good overall match"

# Test function
if __name__ == "__main__":
    # This would be tested with actual DataFrame
    print("HybridRanker ready for integration")