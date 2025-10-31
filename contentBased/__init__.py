"""
Content-Based Filtering Module for Twitch Streamer Recommendations

This module provides content-based filtering functionality for recommending
Twitch streamers based on their audience behavior characteristics.
"""

from .content_based_model import ContentBasedRecommender
from .feature_extraction import StreamerFeatureExtractor

__all__ = ['ContentBasedRecommender', 'StreamerFeatureExtractor']
__version__ = '1.0.0'