"""
Content-Based Filtering - Feature Extraction
Extracts streamer features from viewing data for content-based recommendations.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import os


class StreamerFeatureExtractor:
    """Extract features from viewing data to create streamer profiles."""

    def __init__(self, data_path):
        """
        Initialize the feature extractor.

        Args:
            data_path: Path to the CSV file containing viewing data
        """
        self.data_path = data_path
        self.data = None
        self.streamer_features = None

    def load_data(self):
        """Load the viewing data from CSV."""
        print("Loading data...")
        self.data = pd.read_csv(
            self.data_path,
            names=['user_id', 'stream_id', 'streamer_name', 'time_start', 'time_stop']
        )
        print(f"Loaded {len(self.data)} viewing records")
        print(f"Unique users: {self.data['user_id'].nunique()}")
        print(f"Unique streamers: {self.data['streamer_name'].nunique()}")

    def extract_features(self):
        """
        Extract features for each streamer:
        - Total viewership (total watch time)
        - Unique viewer count
        - Average session duration
        - Session frequency (number of sessions)
        - Viewer retention rate
        - Peak activity time
        - Growth trend
        """
        print("\nExtracting streamer features...")

        # Calculate session durations
        self.data['session_duration'] = self.data['time_stop'] - self.data['time_start']

        features = []

        for streamer in self.data['streamer_name'].unique():
            streamer_data = self.data[self.data['streamer_name'] == streamer]

            # Basic metrics
            total_viewership = streamer_data['session_duration'].sum()
            unique_viewers = streamer_data['user_id'].nunique()
            avg_session_duration = streamer_data['session_duration'].mean()
            num_sessions = len(streamer_data)

            # Viewer retention: viewers who watched more than once
            viewer_counts = streamer_data['user_id'].value_counts()
            returning_viewers = (viewer_counts > 1).sum()
            retention_rate = returning_viewers / unique_viewers if unique_viewers > 0 else 0

            # Peak activity time (most common start time)
            peak_time = streamer_data['time_start'].mode()[0] if len(streamer_data) > 0 else 0

            # Growth trend: compare first half vs second half of time period
            mid_time = streamer_data['time_start'].median()
            first_half = streamer_data[streamer_data['time_start'] <= mid_time]
            second_half = streamer_data[streamer_data['time_start'] > mid_time]

            first_half_viewers = first_half['user_id'].nunique() if len(first_half) > 0 else 1
            second_half_viewers = second_half['user_id'].nunique() if len(second_half) > 0 else 1
            growth_rate = (second_half_viewers - first_half_viewers) / first_half_viewers

            # Engagement score: average sessions per unique viewer
            engagement_score = num_sessions / unique_viewers if unique_viewers > 0 else 0

            features.append({
                'streamer_name': streamer,
                'total_viewership': total_viewership,
                'unique_viewers': unique_viewers,
                'avg_session_duration': avg_session_duration,
                'num_sessions': num_sessions,
                'retention_rate': retention_rate,
                'peak_time': peak_time,
                'growth_rate': growth_rate,
                'engagement_score': engagement_score
            })

        self.streamer_features = pd.DataFrame(features)
        print(f"\nExtracted features for {len(self.streamer_features)} streamers")

        return self.streamer_features

    def normalize_features(self):
        """Normalize features for similarity computation."""
        print("\nNormalizing features...")

        # Select numeric columns for normalization
        numeric_cols = [
            'total_viewership', 'unique_viewers', 'avg_session_duration',
            'num_sessions', 'retention_rate', 'peak_time',
            'growth_rate', 'engagement_score'
        ]

        # Create normalized DataFrame
        normalized_data = self.streamer_features.copy()

        for col in numeric_cols:
            min_val = self.streamer_features[col].min()
            max_val = self.streamer_features[col].max()

            if max_val - min_val > 0:
                normalized_data[f'{col}_norm'] = (
                    (self.streamer_features[col] - min_val) / (max_val - min_val)
                )
            else:
                normalized_data[f'{col}_norm'] = 0

        self.streamer_features = normalized_data
        return normalized_data

    def save_features(self, output_dir='contentBased/processed'):
        """Save extracted and normalized features."""
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV
        csv_path = os.path.join(output_dir, 'streamer_features.csv')
        self.streamer_features.to_csv(csv_path, index=False)
        print(f"\nFeatures saved to {csv_path}")

        # Save as pickle for faster loading
        pickle_path = os.path.join(output_dir, 'streamer_features.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.streamer_features, f)
        print(f"Features saved to {pickle_path}")

        return csv_path, pickle_path

    def get_feature_statistics(self):
        """Display statistics about extracted features."""
        print("\n" + "="*60)
        print("STREAMER FEATURE STATISTICS")
        print("="*60)

        stats_cols = [
            'total_viewership', 'unique_viewers', 'avg_session_duration',
            'num_sessions', 'retention_rate', 'engagement_score', 'growth_rate'
        ]

        print(self.streamer_features[stats_cols].describe())

        # Top streamers by various metrics
        print("\n" + "-"*60)
        print("TOP 10 STREAMERS BY TOTAL VIEWERSHIP:")
        print("-"*60)
        top_viewership = self.streamer_features.nlargest(10, 'total_viewership')[
            ['streamer_name', 'total_viewership', 'unique_viewers']
        ]
        print(top_viewership.to_string(index=False))

        print("\n" + "-"*60)
        print("TOP 10 STREAMERS BY ENGAGEMENT SCORE:")
        print("-"*60)
        top_engagement = self.streamer_features.nlargest(10, 'engagement_score')[
            ['streamer_name', 'engagement_score', 'retention_rate']
        ]
        print(top_engagement.to_string(index=False))


def main():
    """Run feature extraction pipeline."""
    # Path to data
    data_path = '../data/100k_a.csv'

    # Initialize extractor
    extractor = StreamerFeatureExtractor(data_path)

    # Load data
    extractor.load_data()

    # Extract features
    extractor.extract_features()

    # Normalize features
    extractor.normalize_features()

    # Display statistics
    extractor.get_feature_statistics()

    # Save features
    extractor.save_features()

    print("\n✓ Feature extraction complete!")


if __name__ == '__main__':
    main()
