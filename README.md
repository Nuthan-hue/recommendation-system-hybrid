# Hybrid Recommendation System - Twitch Streaming Platform

## Project Overview
Building a hybrid recommendation system that combines **collaborative filtering** and **content-based filtering** to recommend Twitch streamers to viewers based on their viewing history and preferences.

## The Story: "The Twitch Discovery Engine"
This project tells the story of how viewers discover new streamers on Twitch by:
- Analyzing viewing patterns across 100k users
- Identifying similar streamers based on audience overlap
- Predicting which streamers a user might enjoy next
- Solving the cold-start problem for new users

## Dataset
**Source**: [UCSD Twitch Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch)
- **File**: `100k_a.csv` (114MB)
- **Records**: ~3M interactions
- **Users**: 100k viewers
- **Streamers**: ~162k streamers
- **Time Period**: 43 days of data captured every 10 minutes

### Data Structure
```
user_id, stream_id, streamer_name, time_start, time_stop
```
- One user watches multiple streamers over time
- Each row represents a viewing session
- Time measured in 10-minute intervals

## Data Preprocessing Strategy

### 1. Collaborative Filtering Preprocessing
**Goal**: Create user-item interaction matrix

**Steps**:
- Build user-streamer interaction matrix (users × streamers)
- Aggregate viewing metrics:
  - Total watch time per user-streamer pair
  - Number of viewing sessions (frequency)
  - Implicit ratings (watch time as proxy for preference)
- Handle sparsity: Most users only watch a small subset of streamers
- Create train/test split for evaluation

**Output**:
- User-item matrix (sparse format)
- User and streamer ID mappings

### 2. Content-Based Filtering Preprocessing
**Goal**: Build streamer feature profiles

**Steps**:
- Extract streamer characteristics:
  - Total viewership (popularity)
  - Unique viewer count (reach)
  - Average session duration (engagement)
  - Peak viewing times (temporal patterns)
  - Viewer retention rate
  - Growth trends over 43 days
- Create streamer feature vectors
- Compute streamer similarity based on audience behavior patterns

**Output**:
- Streamer feature matrix
- Streamer profile metadata

### 3. Hybrid Approach
Combine both methods to leverage:
- **Collaborative filtering**: "Users who watched X also watched Y"
- **Content-based filtering**: "Streamers similar to X based on audience characteristics"

## Directory Structure
```
hybrid_reccomendations/
├── data/                    # Raw datasets
├── collaborative/           # Collaborative filtering implementation
├── contentBased/           # Content-based filtering implementation
├── README.md               # This file
└── data_preprocessing.py   # Preprocessing scripts (coming soon)
```

## Next Steps
1. Data exploration and statistics
2. Implement preprocessing pipelines
3. Build collaborative filtering model
4. Build content-based filtering model
5. Combine into hybrid recommender
6. Evaluate and visualize results