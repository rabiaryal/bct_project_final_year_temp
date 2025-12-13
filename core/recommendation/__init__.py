"""
Recommendation Module
Neural dual-encoder and hybrid ranking for college recommendations
"""

from .neural_recommender import create_neural_recommender, NeuralRecommender
from .hybrid_ranker import HybridRanker

__all__ = ['create_neural_recommender', 'NeuralRecommender', 'HybridRanker']
