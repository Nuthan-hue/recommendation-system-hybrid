# Content-Based Filtering for Twitch Streamer Recommendations

This module implements content-based filtering to recommend Twitch streamers based on their audience behavior characteristics and feature similarity.

## Overview

Content-based filtering recommends streamers by analyzing their intrinsic features and finding similar streamers. Unlike collaborative filtering which uses user-streamer interactions, content-based filtering focuses on the characteristics of streamers themselves.

## Features Extracted

For each streamer, we extract the following features:

1. **Total Viewership**: Total watch time across all viewers
2. **Unique Viewers**: Number of distinct viewers
3. **Average Session Duration**: Mean length of viewing sessions
4. **Number of Sessions**: Total viewing sessions
5. **Retention Rate**: Percentage of viewers who return multiple times
6. **Peak Activity Time**: Most popular streaming time
7. **Growth Rate**: Viewer growth trend over time
8. **Engagement Score**: Average sessions per unique viewer

## How It Works

1. **Feature Extraction**: Process raw viewing data to compute streamer features
2. **Normalization**: Normalize features to 0-1 scale for fair comparison
3. **Similarity Computation**: Calculate cosine similarity between all streamer feature vectors
4. **Recommendation Generation**: Find streamers most similar to user's viewing history

## Files

- `feature_extraction.py` - Extract and normalize streamer features from viewing data
- `content_based_model.py` - Content-based recommendation model implementation
- `demo.py` - Interactive demo showcasing the recommendation system
- `requirements.txt` - Python dependencies

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Extract Features

Run feature extraction on the Twitch dataset:

```bash
cd contentBased
python feature_extraction.py
```

This will:
- Load the viewing data from `../data/100k_a.csv`
- Extract features for each streamer
- Normalize features
- Save results to `processed/streamer_features.csv` and `.pkl`

### Step 2: Train Model

Build the similarity matrix:

```bash
python content_based_model.py
```

This will:
- Load the extracted features
- Compute pairwise cosine similarity between all streamers
- Save the trained model to `models/content_model.pkl`

### Step 3: Run Demo

Try out the recommendation system:

```bash
python demo.py
```

This demonstrates:
- Finding similar streamers
- User-based recommendations
- Weighted recommendations
- Streamer profile viewing
- Interactive query mode

## API Usage

### Basic Example

```python
from content_based_model import ContentBasedRecommender

# Load the trained model
recommender = ContentBasedRecommender()
recommender.load_model('models/content_model.pkl')

# Find similar streamers
similar = recommender.get_similar_streamers('streamer_name', top_k=10)
for streamer, score in similar:
    print(f"{streamer}: {score:.4f}")
```

### User Recommendations

```python
# User's viewing history
user_history = ['streamer1', 'streamer2', 'streamer3']

# Get recommendations
recommendations = recommender.recommend_for_user(user_history, top_k=10)
for streamer, score in recommendations:
    print(f"{streamer}: {score:.4f}")
```

### Weighted Recommendations

```python
# Weighted history (streamer, watch_time_weight)
weighted_history = [
    ('streamer1', 5.0),  # Watched most
    ('streamer2', 3.0),
    ('streamer3', 1.0)
]

recommendations = recommender.recommend_for_user(
    weighted_history,
    top_k=10,
    aggregation='weighted_avg'
)
```

### Compare Streamers

```python
# Compare feature profiles
recommender.compare_streamers('streamer1', 'streamer2')
```

## Advantages

- **No Cold Start Problem**: Can recommend streamers even for new users
- **Transparency**: Recommendations based on interpretable features
- **Diversity**: Can find niche streamers with similar characteristics
- **Independence**: Doesn't require large user-item interaction matrix

## Limitations

- **Over-specialization**: May recommend too similar content (filter bubble)
- **No Discovery**: Cannot recommend based on community preferences
- **Feature Dependence**: Quality depends on feature engineering
- **Static**: Doesn't adapt to changing user preferences without retraining

## Next Steps

- Combine with collaborative filtering for hybrid recommendations
- Add more features (game categories, streaming times, language)
- Implement A/B testing framework
- Add real-time feature updates

## Directory Structure

```
contentBased/
├── feature_extraction.py      # Feature extraction pipeline
├── content_based_model.py     # Recommendation model
├── demo.py                    # Interactive demo
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── processed/                 # Extracted features (generated)
│   ├── streamer_features.csv
│   └── streamer_features.pkl
└── models/                    # Trained models (generated)
    └── content_model.pkl
```

## Performance Notes

- Feature extraction: ~2-5 minutes for 100k dataset
- Model training: ~30 seconds
- Recommendation generation: <1 second per query
- Memory usage: ~500MB for full similarity matrix

## Contributing

To extend the content-based system:

1. Add new features in `StreamerFeatureExtractor.extract_features()`
2. Update normalization in `normalize_features()`
3. Optionally add feature weights in `compute_similarity_matrix()`
4. Test with `demo.py`