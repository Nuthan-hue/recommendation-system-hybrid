"""
Content-Based Filtering Model
Recommends streamers based on similarity of their content/audience features.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os


class ContentBasedRecommender:
    """Content-based recommendation system using streamer features."""

    def __init__(self, features_path=None):
        """
        Initialize the content-based recommender.

        Args:
            features_path: Path to the streamer features file (CSV or pickle)
        """
        self.features_path = features_path
        self.streamer_features = None
        self.similarity_matrix = None
        self.streamer_to_idx = {}
        self.idx_to_streamer = {}

    def load_features(self, features_path=None):
        """Load precomputed streamer features."""
        if features_path is None:
            features_path = self.features_path

        if features_path is None:
            raise ValueError("No features path provided")

        print(f"Loading features from {features_path}")

        if features_path.endswith('.pkl'):
            with open(features_path, 'rb') as f:
                self.streamer_features = pickle.load(f)
        else:
            self.streamer_features = pd.read_csv(features_path)

        print(f"Loaded features for {len(self.streamer_features)} streamers")

        # Create mappings
        self.streamer_to_idx = {
            streamer: idx
            for idx, streamer in enumerate(self.streamer_features['streamer_name'])
        }
        self.idx_to_streamer = {
            idx: streamer
            for streamer, idx in self.streamer_to_idx.items()
        }

    def compute_similarity_matrix(self, feature_weights=None):
        """
        Compute pairwise similarity between all streamers.

        Args:
            feature_weights: Dict of feature names to weights (default: equal weights)
        """
        print("\nComputing streamer similarity matrix...")

        # Select normalized features for similarity computation
        feature_cols = [
            'total_viewership_norm', 'unique_viewers_norm',
            'avg_session_duration_norm', 'num_sessions_norm',
            'retention_rate_norm', 'engagement_score_norm',
            'growth_rate_norm', 'peak_time_norm'
        ]

        # Default: equal weights
        if feature_weights is None:
            feature_weights = {col.replace('_norm', ''): 1.0 for col in feature_cols}

        # Apply weights
        weighted_features = self.streamer_features[feature_cols].copy()
        for col in feature_cols:
            base_col = col.replace('_norm', '')
            if base_col in feature_weights:
                weighted_features[col] = weighted_features[col] * feature_weights[base_col]

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(weighted_features)

        print(f"Computed {self.similarity_matrix.shape[0]}x{self.similarity_matrix.shape[1]} similarity matrix")

        return self.similarity_matrix

    def get_similar_streamers(self, streamer_name, top_k=10, return_scores=True):
        """
        Get the most similar streamers to a given streamer.

        Args:
            streamer_name: Name of the streamer
            top_k: Number of similar streamers to return
            return_scores: Whether to return similarity scores

        Returns:
            List of similar streamer names (and optionally scores)
        """
        if streamer_name not in self.streamer_to_idx:
            print(f"Streamer '{streamer_name}' not found in database")
            return []

        streamer_idx = self.streamer_to_idx[streamer_name]
        similarities = self.similarity_matrix[streamer_idx]

        # Get top-k most similar (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]

        if return_scores:
            results = [
                (self.idx_to_streamer[idx], similarities[idx])
                for idx in similar_indices
            ]
        else:
            results = [self.idx_to_streamer[idx] for idx in similar_indices]

        return results

    def recommend_for_user(self, user_history, top_k=10, aggregation='weighted_avg'):
        """
        Recommend streamers for a user based on their viewing history.

        Args:
            user_history: List of (streamer_name, weight) tuples or just streamer names
            top_k: Number of recommendations to return
            aggregation: How to aggregate similarities ('weighted_avg' or 'max')

        Returns:
            List of recommended streamer names with scores
        """
        # Normalize user history format
        if isinstance(user_history[0], tuple):
            history = user_history
        else:
            # Equal weights for all streamers in history
            history = [(s, 1.0) for s in user_history]

        # Filter out streamers not in our database
        valid_history = [
            (s, w) for s, w in history
            if s in self.streamer_to_idx
        ]

        if not valid_history:
            print("No valid streamers in user history")
            return []

        # Aggregate similarity scores
        all_scores = np.zeros(len(self.streamer_to_idx))

        for streamer_name, weight in valid_history:
            streamer_idx = self.streamer_to_idx[streamer_name]
            similarities = self.similarity_matrix[streamer_idx]

            if aggregation == 'weighted_avg':
                all_scores += similarities * weight
            elif aggregation == 'max':
                all_scores = np.maximum(all_scores, similarities * weight)

        # Normalize by number of streamers in history
        if aggregation == 'weighted_avg':
            total_weight = sum(w for _, w in valid_history)
            all_scores /= total_weight

        # Exclude streamers already in history
        watched_indices = [self.streamer_to_idx[s] for s, _ in valid_history]
        all_scores[watched_indices] = -1

        # Get top-k recommendations
        top_indices = np.argsort(all_scores)[::-1][:top_k]
        recommendations = [
            (self.idx_to_streamer[idx], all_scores[idx])
            for idx in top_indices
            if all_scores[idx] > 0
        ]

        return recommendations

    def get_streamer_profile(self, streamer_name):
        """Get the feature profile of a streamer."""
        if streamer_name not in self.streamer_to_idx:
            print(f"Streamer '{streamer_name}' not found")
            return None

        idx = self.streamer_to_idx[streamer_name]
        profile = self.streamer_features.iloc[idx]

        return profile

    def compare_streamers(self, streamer1, streamer2):
        """
        Compare two streamers and show their similarity score and feature differences.

        Args:
            streamer1: First streamer name
            streamer2: Second streamer name
        """
        if streamer1 not in self.streamer_to_idx or streamer2 not in self.streamer_to_idx:
            print("One or both streamers not found")
            return

        idx1 = self.streamer_to_idx[streamer1]
        idx2 = self.streamer_to_idx[streamer2]

        similarity = self.similarity_matrix[idx1, idx2]

        print(f"\n{'='*60}")
        print(f"COMPARISON: {streamer1} vs {streamer2}")
        print(f"{'='*60}")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"\n{'Feature':<25} {streamer1:<15} {streamer2:<15}")
        print("-"*60)

        comparison_cols = [
            'total_viewership', 'unique_viewers', 'avg_session_duration',
            'retention_rate', 'engagement_score', 'growth_rate'
        ]

        profile1 = self.streamer_features.iloc[idx1]
        profile2 = self.streamer_features.iloc[idx2]

        for col in comparison_cols:
            val1 = profile1[col]
            val2 = profile2[col]
            print(f"{col:<25} {val1:<15.2f} {val2:<15.2f}")

    def save_model(self, output_path='contentBased/models/content_model.pkl'):
        """Save the trained model."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'streamer_to_idx': self.streamer_to_idx,
            'idx_to_streamer': self.idx_to_streamer,
            'streamer_features': self.streamer_features
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to {output_path}")

    def load_model(self, model_path='contentBased/models/content_model.pkl'):
        """Load a trained model."""
        print(f"Loading model from {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.similarity_matrix = model_data['similarity_matrix']
        self.streamer_to_idx = model_data['streamer_to_idx']
        self.idx_to_streamer = model_data['idx_to_streamer']
        self.streamer_features = model_data['streamer_features']

        print(f"Model loaded successfully ({len(self.streamer_to_idx)} streamers)")


def main():
    """Train and test the content-based model."""
    # Initialize recommender
    recommender = ContentBasedRecommender()

    # Load features
    features_path = 'contentBased/processed/streamer_features.pkl'
    if not os.path.exists(features_path):
        features_path = 'contentBased/processed/streamer_features.csv'

    recommender.load_features(features_path)

    # Compute similarity matrix
    recommender.compute_similarity_matrix()

    # Save model
    recommender.save_model()

    print("\n✓ Content-based model trained and saved!")


if __name__ == '__main__':
    main()