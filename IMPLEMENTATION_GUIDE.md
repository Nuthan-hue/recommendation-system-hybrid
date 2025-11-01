# Hybrid Recommendation System - Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Content-Based Filtering](#content-based-filtering)
3. [Collaborative Filtering](#collaborative-filtering)
4. [Hybrid Approach](#hybrid-approach)
5. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

### What is a Recommendation System?

A recommendation system predicts what items (in our case, Twitch streamers) a user might be interested in based on various data sources.

### Three Main Approaches:

1. **Content-Based Filtering**: Recommends items similar to what user liked before
   - Uses item features/characteristics
   - "Users who liked X will like similar items"

2. **Collaborative Filtering**: Recommends based on what similar users liked
   - Uses user-item interactions
   - "Users similar to you also liked..."

3. **Hybrid**: Combines both approaches
   - Leverages strengths of both methods
   - Overcomes individual limitations

---

## Content-Based Filtering

### Core Concept
Recommend streamers that are **similar** to streamers a user has already watched, based on the streamers' characteristics (features).

### The Process

#### PHASE 1: FEATURE EXTRACTION

**Goal:** Create a "profile" for each streamer based on their characteristics

##### Step 1.1: Data Preparation
```python
# Input: Raw viewing data (3M rows)
# Columns: user_id, streamSession_id, streamer_name, time_start, time_stop

# Calculate derived columns
df['duration_minutes'] = (df['time_stop'] - df['time_start']) * 10
df['day'] = df['time_start'] // 144  # 144 intervals per day
df['hour_of_day'] = ((df['time_start'] % 144) * 10) // 60
```

**Why?**
- `duration_minutes`: Measures engagement (how long viewers watch)
- `day`: Tracks when streamer was active
- `hour_of_day`: Identifies streaming schedule patterns

---

##### Step 1.2: Aggregate by Streamer
```python
streamer_features = df.groupby('streamer_name').agg({
    'user_id': 'nunique',                    # How many unique viewers?
    'streamSession_id': 'nunique',           # How many streaming sessions?
    'duration_minutes': ['sum', 'mean'],     # Total & average watch time
    'day': ['min', 'max'],                   # Active period
    'hour_of_day': lambda x: x.mode()[0]     # Most common streaming hour
})
```

**Result:** Transform from 3M viewing sessions → 162K streamer profiles

**Example Output:**
```
streamer_name    unique_viewers  num_sessions  total_watch_time  avg_session_duration
mithrain         25000           59            500000            35.2
alptv            8000            23            120000            28.5
```

---

##### Step 1.3: Calculate Advanced Features

```python
# 1. Retention Rate: Do viewers come back?
streamer_features['retention_rate'] = (
    streamer_features['num_sessions'] / streamer_features['unique_viewers']
)

# 2. Engagement Score: Average sessions per viewer
streamer_features['engagement_score'] = (
    streamer_features['num_sessions'] / streamer_features['unique_viewers']
)

# 3. Popularity: Total viewership
streamer_features['popularity'] = streamer_features['total_watch_time']

# 4. Active Days: How long were they streaming?
streamer_features['active_days'] = (
    streamer_features['last_day'] - streamer_features['first_day'] + 1
)

# 5. Growth Rate: Early vs Late viewership
early_period = df[df['day'] <= 14]  # First 2 weeks
late_period = df[df['day'] >= 29]   # Last 2 weeks

early_views = early_period.groupby('streamer_name')['user_id'].nunique()
late_views = late_period.groupby('streamer_name')['user_id'].nunique()

streamer_features['growth_rate'] = (
    (late_views - early_views) / (early_views + 1)  # +1 to avoid division by zero
)

# 6. Consistency: How regularly do they stream?
streamer_features['consistency'] = (
    streamer_features['num_sessions'] / streamer_features['active_days']
)

# 7. Average Viewers per Session
streamer_features['avg_viewers_per_session'] = (
    streamer_features['unique_viewers'] / streamer_features['num_sessions']
)
```

**Feature Summary:**

| Feature | What it measures | Why it matters |
|---------|------------------|----------------|
| `unique_viewers` | Reach | How many people watch this streamer? |
| `num_sessions` | Activity level | How often do they stream? |
| `total_watch_time` | Overall popularity | Total viewing time across all users |
| `avg_session_duration` | Engagement | How long do people watch per session? |
| `retention_rate` | Loyalty | Do viewers return? (>1 = yes) |
| `popularity` | Total viewership | Raw popularity metric |
| `active_days` | Persistence | How long have they been streaming? |
| `growth_rate` | Trending | Growing or declining? |
| `consistency` | Regularity | How often they stream per day |
| `peak_hour` | Schedule pattern | When do they stream? |

---

##### Step 1.4: Normalize Features (Critical!)

**Why normalize?**
- Features have different scales:
  - `total_watch_time`: Can be millions
  - `retention_rate`: Usually 0-10
  - Without normalization, large-scale features dominate similarity calculations

```python
from sklearn.preprocessing import MinMaxScaler

# Select features to use for similarity
feature_columns = [
    'unique_viewers',
    'num_sessions',
    'total_watch_time',
    'avg_session_duration',
    'retention_rate',
    'growth_rate',
    'consistency',
    'peak_hour'
]

# Normalize to 0-1 scale
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(streamer_features[feature_columns])

# Create normalized dataframe
streamer_features_normalized = pd.DataFrame(
    features_normalized,
    columns=feature_columns,
    index=streamer_features.index
)
```

**Before normalization:**
```
total_watch_time: 50000, retention_rate: 1.5
total_watch_time: 5000000, retention_rate: 1.8
```

**After normalization:**
```
total_watch_time: 0.01, retention_rate: 0.45
total_watch_time: 1.0, retention_rate: 0.54
```

---

##### Step 1.5: Save Processed Features

```python
import pickle

# Save as CSV for inspection
streamer_features.to_csv('contentBased/processed/streamer_features.csv')

# Save as pickle for fast loading
with open('contentBased/processed/streamer_features.pkl', 'wb') as f:
    pickle.dump({
        'features': streamer_features,
        'features_normalized': streamer_features_normalized,
        'scaler': scaler,
        'feature_columns': feature_columns
    }, f)
```

---

#### PHASE 2: BUILD SIMILARITY MODEL

**Goal:** Calculate how similar each streamer is to every other streamer

##### Step 2.1: Compute Cosine Similarity

**What is Cosine Similarity?**
- Measures the angle between two feature vectors
- Range: 0 (completely different) to 1 (identical)
- Ignores magnitude, focuses on direction (pattern similarity)

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise similarity
similarity_matrix = cosine_similarity(features_normalized)

# Result: 162,625 × 162,625 matrix
# similarity_matrix[i][j] = similarity between streamer i and j
```

**Example:**
```
              mithrain  alptv  wtcn
mithrain      1.00      0.85   0.32
alptv         0.85      1.00   0.41
wtcn          0.32      0.41   1.00
```

**Interpretation:**
- `mithrain` and `alptv` are very similar (0.85)
- `mithrain` and `wtcn` are less similar (0.32)

---

##### Step 2.2: Create Lookup Structures

```python
# Map streamer names to indices and vice versa
streamer_to_idx = {name: idx for idx, name in enumerate(streamer_features.index)}
idx_to_streamer = {idx: name for name, idx in streamer_to_idx.items()}
```

---

##### Step 2.3: Build Recommendation Function

```python
def get_similar_streamers(streamer_name, top_k=10):
    """
    Find streamers most similar to the given streamer

    Args:
        streamer_name: Name of the streamer
        top_k: Number of recommendations to return

    Returns:
        List of (streamer_name, similarity_score) tuples
    """
    # Get the index of the streamer
    idx = streamer_to_idx[streamer_name]

    # Get similarity scores with all other streamers
    similarity_scores = similarity_matrix[idx]

    # Get indices of top_k most similar (excluding the streamer itself)
    # argsort returns indices that would sort the array
    # [-top_k-1:-1] gets the top k (excluding the last which is the streamer itself)
    # [::-1] reverses to get descending order
    similar_indices = similarity_scores.argsort()[-top_k-1:-1][::-1]

    # Return streamer names and scores
    return [(idx_to_streamer[i], similarity_scores[i]) for i in similar_indices]
```

**Example Usage:**
```python
similar = get_similar_streamers('mithrain', top_k=5)
# Output:
# [('alptv', 0.89),
#  ('wtcn', 0.87),
#  ('elraenn', 0.85),
#  ('kendinemuzisyen', 0.83),
#  ('jahrein', 0.81)]
```

---

##### Step 2.4: User-Based Recommendations

```python
def recommend_for_user(user_history, top_k=10, aggregation='mean'):
    """
    Recommend streamers based on user's viewing history

    Args:
        user_history: List of streamer names the user watched
        top_k: Number of recommendations
        aggregation: How to combine scores ('mean', 'max', 'weighted_avg')

    Returns:
        List of recommended streamers with scores
    """
    # Initialize score accumulator
    recommendation_scores = np.zeros(len(streamer_to_idx))

    # For each streamer in user's history
    for streamer in user_history:
        if streamer in streamer_to_idx:
            idx = streamer_to_idx[streamer]
            # Add similarity scores
            recommendation_scores += similarity_matrix[idx]

    # Average the scores
    if aggregation == 'mean':
        recommendation_scores /= len(user_history)

    # Remove streamers already watched
    for streamer in user_history:
        if streamer in streamer_to_idx:
            idx = streamer_to_idx[streamer]
            recommendation_scores[idx] = -1  # Mark as already watched

    # Get top recommendations
    top_indices = recommendation_scores.argsort()[-top_k:][::-1]

    return [(idx_to_streamer[i], recommendation_scores[i]) for i in top_indices]
```

**Example Usage:**
```python
user_history = ['mithrain', 'alptv', 'wtcn']
recommendations = recommend_for_user(user_history, top_k=10)

# Output:
# [('elraenn', 0.82),
#  ('jahrein', 0.79),
#  ('kendinemuzisyen', 0.76),
#  ...]
```

---

##### Step 2.5: Weighted Recommendations (Advanced)

```python
def recommend_weighted(user_history_weighted, top_k=10):
    """
    Weighted recommendations based on watch time

    Args:
        user_history_weighted: List of (streamer_name, weight) tuples
        Example: [('mithrain', 500), ('alptv', 200)]  # watch time in minutes

    Returns:
        Weighted recommendations
    """
    recommendation_scores = np.zeros(len(streamer_to_idx))
    total_weight = 0

    for streamer, weight in user_history_weighted:
        if streamer in streamer_to_idx:
            idx = streamer_to_idx[streamer]
            # Weight by how much the user watched this streamer
            recommendation_scores += similarity_matrix[idx] * weight
            total_weight += weight

    # Normalize by total weight
    recommendation_scores /= total_weight

    # Remove already watched
    for streamer, _ in user_history_weighted:
        if streamer in streamer_to_idx:
            recommendation_scores[streamer_to_idx[streamer]] = -1

    top_indices = recommendation_scores.argsort()[-top_k:][::-1]
    return [(idx_to_streamer[i], recommendation_scores[i]) for i in top_indices]
```

---

##### Step 2.6: Save the Model

```python
import pickle

# Save everything needed for inference
model = {
    'similarity_matrix': similarity_matrix,
    'streamer_to_idx': streamer_to_idx,
    'idx_to_streamer': idx_to_streamer,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'streamer_features': streamer_features
}

with open('contentBased/models/content_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved! Size: {similarity_matrix.shape}")
```

---

#### PHASE 3: EVALUATION

##### Method 1: Sanity Checks
```python
# Test similar streamers
print("Similar to 'mithrain':")
for streamer, score in get_similar_streamers('mithrain', top_k=5):
    print(f"  {streamer}: {score:.3f}")
```

##### Method 2: Compare Features
```python
def compare_streamers(streamer1, streamer2):
    """Compare feature profiles of two streamers"""
    features1 = streamer_features.loc[streamer1]
    features2 = streamer_features.loc[streamer2]

    print(f"\nComparing {streamer1} vs {streamer2}:\n")
    for feature in feature_columns:
        val1 = features1[feature]
        val2 = features2[feature]
        print(f"{feature:25} {val1:10.2f} vs {val2:10.2f}")
```

---

### Content-Based Filtering: Advantages & Disadvantages

#### ✅ Advantages:
1. **No Cold Start Problem (for items)**: Can recommend new streamers immediately if we have their features
2. **Transparency**: Can explain WHY a streamer was recommended based on features
3. **User Independence**: Doesn't need data from other users
4. **Diversity**: Can find niche streamers with similar characteristics

#### ❌ Disadvantages:
1. **Cold Start (for users)**: Need user history to make recommendations
2. **Over-specialization**: May recommend too similar content (filter bubble)
3. **Feature Engineering**: Quality depends heavily on feature selection
4. **Limited Serendipity**: Cannot discover unexpected but good recommendations
5. **No Community Wisdom**: Ignores what other users think

---

## Collaborative Filtering

### Core Concept
Recommend streamers based on what **similar users** watched, ignoring streamer characteristics.

### Two Main Types:

1. **User-Based**: Find similar users, recommend what they watched
2. **Item-Based**: Find similar items based on co-watching patterns
3. **Matrix Factorization**: Decompose user-item matrix to find latent factors

---

### The Process

#### PHASE 1: BUILD USER-ITEM INTERACTION MATRIX

**Goal:** Create a matrix where rows = users, columns = streamers, values = interaction strength

##### Step 1.1: Aggregate Interactions

```python
# Calculate total watch time per user-streamer pair
user_streamer_matrix = df.groupby(['user_id', 'streamer_name'])['duration_minutes'].sum()

# Convert to wide format (matrix)
interaction_matrix = user_streamer_matrix.unstack(fill_value=0)

# Result: 100,000 users × 162,625 streamers
# Very sparse matrix (most values are 0)
```

**Example:**
```
              mithrain  alptv  wtcn  ...
user_id
1             520       90     0     ...
2             0         0      180   ...
3             340       120    60    ...
```

**Interpretation:**
- User 1 watched mithrain for 520 minutes total
- User 2 didn't watch mithrain or alptv
- User 3 watched all three

---

##### Step 1.2: Implicit vs Explicit Ratings

**Implicit Feedback** (what we have):
- Watch time, clicks, views
- No explicit rating (1-5 stars)
- More data, but noisy

**Convert to Implicit Ratings:**
```python
# Method 1: Binary (watched or not)
interaction_binary = (interaction_matrix > 0).astype(int)

# Method 2: Normalized watch time
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
interaction_normalized = scaler.fit_transform(interaction_matrix)

# Method 3: Log-scaled (reduces impact of very long watches)
interaction_log = np.log1p(interaction_matrix)  # log(1 + x)
```

---

##### Step 1.3: Handle Sparsity

**Problem:** Matrix is ~99.9% sparse (most users watched <1% of streamers)

**Solutions:**

**Option 1: Use only active users/popular streamers**
```python
# Filter users with at least 5 streamers watched
active_users = (interaction_matrix > 0).sum(axis=1) >= 5
interaction_matrix_filtered = interaction_matrix[active_users]

# Filter streamers with at least 100 viewers
popular_streamers = (interaction_matrix > 0).sum(axis=0) >= 100
interaction_matrix_filtered = interaction_matrix_filtered[popular_streamers]
```

**Option 2: Use sparse matrix format**
```python
from scipy.sparse import csr_matrix
interaction_sparse = csr_matrix(interaction_matrix.values)
# Saves memory: only stores non-zero values
```

---

#### PHASE 2: COLLABORATIVE FILTERING APPROACHES

### Approach A: User-Based Collaborative Filtering

**Idea:** "Users similar to you also watched..."

##### Step 1: Compute User Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise user similarity
user_similarity = cosine_similarity(interaction_matrix)

# Result: 100,000 × 100,000 matrix
# user_similarity[i][j] = similarity between user i and j
```

##### Step 2: Find Similar Users

```python
def find_similar_users(user_id, top_k=10):
    """Find k most similar users"""
    user_idx = user_id_to_idx[user_id]
    similarity_scores = user_similarity[user_idx]
    similar_indices = similarity_scores.argsort()[-top_k-1:-1][::-1]
    return [(idx_to_user_id[i], similarity_scores[i]) for i in similar_indices]
```

##### Step 3: Generate Recommendations

```python
def recommend_user_based(user_id, top_k=10, n_neighbors=20):
    """
    Recommend based on similar users

    Args:
        user_id: Target user
        top_k: Number of recommendations
        n_neighbors: Number of similar users to consider
    """
    # Find similar users
    similar_users = find_similar_users(user_id, n_neighbors)

    # Get user's current viewing history
    user_watched = set(interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index)

    # Aggregate recommendations from similar users
    recommendation_scores = {}

    for similar_user_id, similarity_score in similar_users:
        # Get what this similar user watched
        similar_user_watched = interaction_matrix.loc[similar_user_id]

        # Add weighted scores
        for streamer, watch_time in similar_user_watched.items():
            if watch_time > 0 and streamer not in user_watched:
                if streamer not in recommendation_scores:
                    recommendation_scores[streamer] = 0
                recommendation_scores[streamer] += watch_time * similarity_score

    # Sort and return top k
    sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_recs[:top_k]
```

---

### Approach B: Item-Based Collaborative Filtering

**Idea:** "Streamers watched together by similar audiences"

##### Step 1: Compute Streamer Similarity

```python
# Transpose matrix: now rows = streamers, columns = users
streamer_similarity = cosine_similarity(interaction_matrix.T)

# Result: 162,625 × 162,625 matrix
# streamer_similarity[i][j] = how often streamers i and j are watched together
```

**Key Difference from Content-Based:**
- Content-based: Similarity based on features (popularity, genre, etc.)
- Collaborative: Similarity based on co-watching patterns

##### Step 2: Generate Recommendations

```python
def recommend_item_based(user_id, top_k=10):
    """Recommend based on item similarity"""
    # Get streamers user has watched
    user_watched = interaction_matrix.loc[user_id]
    watched_streamers = user_watched[user_watched > 0].index

    # Calculate scores
    recommendation_scores = np.zeros(len(streamer_to_idx))

    for streamer in watched_streamers:
        idx = streamer_to_idx[streamer]
        watch_time = user_watched[streamer]

        # Add weighted similarity scores
        recommendation_scores += streamer_similarity[idx] * watch_time

    # Remove already watched
    for streamer in watched_streamers:
        recommendation_scores[streamer_to_idx[streamer]] = -1

    # Get top k
    top_indices = recommendation_scores.argsort()[-top_k:][::-1]
    return [(idx_to_streamer[i], recommendation_scores[i]) for i in top_indices]
```

---

### Approach C: Matrix Factorization (Advanced)

**Idea:** Decompose the user-item matrix into lower-dimensional representations

##### Method 1: SVD (Singular Value Decomposition)

```python
from scipy.sparse.linalg import svds

# Perform SVD
# R ≈ U × Σ × V^T
# R: user-item matrix (m × n)
# U: user factors (m × k)
# Σ: singular values (k × k)
# V: item factors (n × k)

k = 50  # Number of latent factors
U, sigma, Vt = svds(interaction_sparse, k=k)

# Reconstruct predicted ratings
sigma_matrix = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma_matrix), Vt)
```

**What are latent factors?**
Hidden features that explain viewing patterns:
- Factor 1: "Action game streamers"
- Factor 2: "Late night streamers"
- Factor 3: "Tournament players"
- etc.

##### Method 2: Alternating Least Squares (ALS)

```python
from implicit.als import AlternatingLeastSquares

# Better for implicit feedback (watch time, not ratings)
model = AlternatingLeastSquares(factors=50, iterations=15)

# Fit model
model.fit(interaction_sparse.T)  # Transpose: items × users

# Generate recommendations
def recommend_als(user_id, top_k=10):
    user_idx = user_id_to_idx[user_id]
    recommendations = model.recommend(user_idx, interaction_sparse[user_idx], N=top_k)
    return [(idx_to_streamer[idx], score) for idx, score in recommendations]
```

---

##### Method 3: Neural Collaborative Filtering

```python
import tensorflow as tf
from tensorflow.keras import layers

# Build neural network for collaborative filtering
def build_ncf_model(n_users, n_items, embedding_dim=50):
    # Input layers
    user_input = layers.Input(shape=(1,))
    item_input = layers.Input(shape=(1,))

    # Embedding layers
    user_embedding = layers.Embedding(n_users, embedding_dim)(user_input)
    item_embedding = layers.Embedding(n_items, embedding_dim)(item_input)

    # Flatten
    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)

    # Concatenate
    concat = layers.Concatenate()([user_vec, item_vec])

    # Dense layers
    dense1 = layers.Dense(128, activation='relu')(concat)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output = layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train model
model = build_ncf_model(n_users=100000, n_items=162625)
# Prepare training data...
# model.fit(...)
```

---

#### PHASE 3: EVALUATION

##### Split Data: Train/Test

```python
from sklearn.model_selection import train_test_split

# For each user, hold out some streamers for testing
def train_test_split_per_user(interaction_matrix, test_ratio=0.2):
    train_matrix = interaction_matrix.copy()
    test_matrix = pd.DataFrame(0, index=interaction_matrix.index, columns=interaction_matrix.columns)

    for user in interaction_matrix.index:
        # Get streamers user watched
        watched = interaction_matrix.loc[user][interaction_matrix.loc[user] > 0]

        if len(watched) > 5:  # Only if user watched enough
            # Sample test streamers
            test_streamers = watched.sample(frac=test_ratio).index

            # Move to test set
            test_matrix.loc[user, test_streamers] = interaction_matrix.loc[user, test_streamers]
            train_matrix.loc[user, test_streamers] = 0

    return train_matrix, test_matrix
```

##### Metrics

**1. Precision@K**
```python
def precision_at_k(recommended, actual, k):
    """
    Precision@K: Of the top K recommendations, how many were actually watched?
    """
    recommended_k = set(recommended[:k])
    actual_set = set(actual)
    return len(recommended_k & actual_set) / k
```

**2. Recall@K**
```python
def recall_at_k(recommended, actual, k):
    """
    Recall@K: Of all items user watched, how many were in top K recommendations?
    """
    recommended_k = set(recommended[:k])
    actual_set = set(actual)
    return len(recommended_k & actual_set) / len(actual_set)
```

**3. Mean Average Precision (MAP)**
```python
def average_precision(recommended, actual):
    """Average precision for one user"""
    score = 0.0
    num_hits = 0.0

    for i, rec in enumerate(recommended):
        if rec in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(actual)

def mean_average_precision(all_recommendations, all_actuals):
    """MAP across all users"""
    return np.mean([average_precision(rec, act)
                    for rec, act in zip(all_recommendations, all_actuals)])
```

**4. NDCG (Normalized Discounted Cumulative Gain)**
```python
from sklearn.metrics import ndcg_score

def ndcg_at_k(recommended, actual, k):
    """
    NDCG@K: Considers both relevance and ranking position
    """
    # Create relevance scores (1 if in actual, 0 otherwise)
    relevance = [1 if item in actual else 0 for item in recommended[:k]]

    # Ideal ranking (all relevant items first)
    ideal = sorted(relevance, reverse=True)

    if sum(ideal) == 0:
        return 0.0

    return ndcg_score([ideal], [relevance])
```

---

### Collaborative Filtering: Advantages & Disadvantages

#### ✅ Advantages:
1. **No Feature Engineering**: Works with just interaction data
2. **Discovers Unexpected Patterns**: Can find non-obvious connections
3. **Community Wisdom**: Leverages collective user behavior
4. **Works Across Domains**: Same approach works for any recommendation task

#### ❌ Disadvantages:
1. **Cold Start Problem**:
   - New users: No history to base recommendations on
   - New items: No interactions to learn from
2. **Sparsity**: Most user-item pairs have no interaction
3. **Scalability**: Computing similarity for millions of users is expensive
4. **Popularity Bias**: Tends to recommend popular items
5. **No Transparency**: Hard to explain why something was recommended

---

## Hybrid Approach

### Why Hybrid?

Combine strengths, mitigate weaknesses:

| Method | Strength | Weakness |
|--------|----------|----------|
| Content-Based | Works for new items, transparent | Filter bubble, needs features |
| Collaborative | Discovers patterns, community wisdom | Cold start, sparsity |
| **Hybrid** | **Best of both** | **More complex** |

---

### Hybrid Strategies

#### Strategy 1: Weighted Combination

```python
def hybrid_weighted(user_id, alpha=0.5, top_k=10):
    """
    Combine content and collaborative scores with weights

    Args:
        user_id: Target user
        alpha: Weight for content-based (1-alpha for collaborative)
        top_k: Number of recommendations
    """
    # Get recommendations from both methods
    content_recs = recommend_content_based(user_id, top_k=100)
    collab_recs = recommend_collaborative(user_id, top_k=100)

    # Normalize scores to 0-1
    content_scores = {name: score for name, score in content_recs}
    collab_scores = {name: score for name, score in collab_recs}

    # Get all candidate streamers
    all_streamers = set(content_scores.keys()) | set(collab_scores.keys())

    # Combine scores
    hybrid_scores = {}
    for streamer in all_streamers:
        content_score = content_scores.get(streamer, 0)
        collab_score = collab_scores.get(streamer, 0)
        hybrid_scores[streamer] = alpha * content_score + (1 - alpha) * collab_score

    # Sort and return
    sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_recs[:top_k]
```

---

#### Strategy 2: Switching

```python
def hybrid_switching(user_id, top_k=10, min_interactions=5):
    """
    Switch between methods based on user profile

    - New users (< min_interactions): Use content-based
    - Active users: Use collaborative
    """
    # Count user's interactions
    n_interactions = (interaction_matrix.loc[user_id] > 0).sum()

    if n_interactions < min_interactions:
        # Cold start: use content-based
        return recommend_content_based(user_id, top_k)
    else:
        # Enough data: use collaborative
        return recommend_collaborative(user_id, top_k)
```

---

#### Strategy 3: Cascade

```python
def hybrid_cascade(user_id, top_k=10, content_threshold=0.8):
    """
    Use content-based first, then fill gaps with collaborative

    1. Get content-based recommendations above threshold
    2. If not enough, add collaborative recommendations
    """
    # Get content recommendations
    content_recs = recommend_content_based(user_id, top_k=50)

    # Filter by threshold
    high_quality = [(name, score) for name, score in content_recs if score >= content_threshold]

    # If we have enough, return
    if len(high_quality) >= top_k:
        return high_quality[:top_k]

    # Otherwise, add collaborative recommendations
    n_needed = top_k - len(high_quality)
    collab_recs = recommend_collaborative(user_id, top_k=n_needed)

    return high_quality + collab_recs[:n_needed]
```

---

#### Strategy 4: Feature Augmentation

```python
def hybrid_feature_augmentation(user_id, top_k=10):
    """
    Use collaborative features as input to content-based model

    Add collaborative signals as features:
    - Average rating from similar users
    - Co-watching patterns
    - Popularity among similar users
    """
    # Get user's similar users
    similar_users = find_similar_users(user_id, top_k=20)

    # For each streamer, compute collaborative features
    collab_features = {}
    for streamer in streamer_features.index:
        # How many similar users watched this?
        similar_user_watches = sum(
            1 for similar_user, _ in similar_users
            if interaction_matrix.loc[similar_user, streamer] > 0
        )

        # Average watch time among similar users
        avg_watch_time = np.mean([
            interaction_matrix.loc[similar_user, streamer]
            for similar_user, _ in similar_users
        ])

        collab_features[streamer] = {
            'similar_user_count': similar_user_watches,
            'avg_similar_watch_time': avg_watch_time
        }

    # Augment streamer features with collaborative features
    # ... then use content-based recommendation with augmented features
```

---

#### Strategy 5: Meta-Learning

```python
def hybrid_meta_learning(user_id, top_k=10):
    """
    Train a model to predict which recommendation method works best

    Features:
    - User activity level
    - User diversity (how many different streamers)
    - Time since first interaction

    Prediction:
    - Which method (content/collab) will perform better
    """
    # Extract user features
    user_features = extract_user_features(user_id)

    # Predict best method
    best_method = meta_model.predict(user_features)

    if best_method == 'content':
        return recommend_content_based(user_id, top_k)
    else:
        return recommend_collaborative(user_id, top_k)
```

---

### Choosing Alpha (Weighted Combination)

**Optimize using validation set:**

```python
def find_optimal_alpha(validation_users, validation_ground_truth):
    """Find best alpha value"""
    best_alpha = 0.5
    best_score = 0

    for alpha in np.arange(0, 1.1, 0.1):
        scores = []
        for user_id in validation_users:
            recs = hybrid_weighted(user_id, alpha=alpha, top_k=10)
            recs_names = [name for name, _ in recs]
            ground_truth = validation_ground_truth[user_id]

            score = precision_at_k(recs_names, ground_truth, k=10)
            scores.append(score)

        avg_score = np.mean(scores)
        print(f"Alpha={alpha:.1f}, Precision@10={avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha

    return best_alpha
```

---

## Evaluation Metrics

### A/B Testing Framework

```python
def ab_test(method_a, method_b, test_users, n_simulations=1000):
    """
    Compare two recommendation methods

    Returns:
    - Win rate for method A
    - Average score difference
    - Statistical significance (p-value)
    """
    wins_a = 0
    wins_b = 0
    ties = 0

    scores_a = []
    scores_b = []

    for user_id in test_users:
        # Get recommendations
        recs_a = method_a(user_id, top_k=10)
        recs_b = method_b(user_id, top_k=10)

        # Get ground truth (test set)
        ground_truth = test_matrix.loc[user_id][test_matrix.loc[user_id] > 0].index

        # Calculate metrics
        score_a = precision_at_k([name for name, _ in recs_a], ground_truth, 10)
        score_b = precision_at_k([name for name, _ in recs_b], ground_truth, 10)

        scores_a.append(score_a)
        scores_b.append(score_b)

        if score_a > score_b:
            wins_a += 1
        elif score_b > score_a:
            wins_b += 1
        else:
            ties += 1

    # Statistical test
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(scores_a, scores_b)

    print(f"Method A wins: {wins_a} ({wins_a/len(test_users)*100:.1f}%)")
    print(f"Method B wins: {wins_b} ({wins_b/len(test_users)*100:.1f}%)")
    print(f"Ties: {ties}")
    print(f"Avg score A: {np.mean(scores_a):.4f}")
    print(f"Avg score B: {np.mean(scores_b):.4f}")
    print(f"P-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
```

---

## Complete Implementation Checklist

### Phase 1: Data Preparation
- [ ] Load raw data (100k_a.csv)
- [ ] Calculate derived features (duration, day, hour)
- [ ] Handle missing values
- [ ] Exploratory data analysis

### Phase 2: Content-Based Filtering
- [ ] Aggregate data by streamer
- [ ] Extract streamer features (8-10 features)
- [ ] Normalize features
- [ ] Compute similarity matrix
- [ ] Build recommendation functions
- [ ] Save content model

### Phase 3: Collaborative Filtering
- [ ] Build user-item interaction matrix
- [ ] Handle sparsity (filtering/sparse format)
- [ ] Choose approach (user-based/item-based/matrix factorization)
- [ ] Compute similarities or train model
- [ ] Build recommendation functions
- [ ] Save collaborative model

### Phase 4: Hybrid System
- [ ] Choose hybrid strategy (weighted/switching/cascade)
- [ ] Implement combination logic
- [ ] Optimize hyperparameters (alpha, thresholds)
- [ ] Save hybrid model

### Phase 5: Evaluation
- [ ] Split data (train/test)
- [ ] Implement metrics (Precision@K, Recall@K, MAP, NDCG)
- [ ] Compare methods (content vs collab vs hybrid)
- [ ] Perform A/B testing
- [ ] Analyze results

### Phase 6: Deployment
- [ ] Create API/demo interface
- [ ] Optimize for speed (caching, indexing)
- [ ] Handle edge cases (new users, new items)
- [ ] Monitor performance
- [ ] Set up A/B testing framework

---

## Recommended Reading

### Papers:
1. "Matrix Factorization Techniques for Recommender Systems" - Koren, Bell, Volinsky
2. "Collaborative Filtering for Implicit Feedback Datasets" - Hu, Koren, Volinsky
3. "Neural Collaborative Filtering" - He et al.

### Libraries:
- `scikit-learn`: Basic ML algorithms
- `surprise`: Recommender systems library
- `implicit`: Fast collaborative filtering for implicit feedback
- `lightfm`: Hybrid recommender (content + collaborative)
- `tensorflow/pytorch`: Deep learning approaches

### Tools:
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scipy`: Sparse matrices
- `matplotlib/seaborn`: Visualization

---

## Next Steps

1. **Start with Content-Based**: Easier to understand and implement
2. **Add Collaborative**: Once content-based works, add collaborative filtering
3. **Build Hybrid**: Combine both methods
4. **Evaluate**: Compare all approaches
5. **Iterate**: Improve based on results

Good luck building your hybrid recommendation system!