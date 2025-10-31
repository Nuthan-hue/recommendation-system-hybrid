"""
Content-Based Filtering Demo
Demonstrates the content-based recommendation system in action.
"""

import os
import sys
from content_based_model import ContentBasedRecommender


def print_recommendations(recommendations, title="Recommendations"):
    """Pretty print recommendations."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    if not recommendations:
        print("No recommendations available")
        return

    print(f"{'Rank':<6} {'Streamer Name':<30} {'Similarity Score':<15}")
    print("-"*70)

    for rank, (streamer, score) in enumerate(recommendations, 1):
        print(f"{rank:<6} {streamer:<30} {score:.4f}")


def demo_similar_streamers(recommender):
    """Demo: Find similar streamers."""
    print("\n" + "="*70)
    print("DEMO 1: FIND SIMILAR STREAMERS")
    print("="*70)

    # Pick a popular streamer to demonstrate
    test_streamer = recommender.idx_to_streamer[0]  # First streamer in database

    print(f"\nFinding streamers similar to: {test_streamer}")

    similar = recommender.get_similar_streamers(test_streamer, top_k=10)
    print_recommendations(similar, f"Streamers Similar to '{test_streamer}'")

    # Show profile comparison
    if similar:
        most_similar = similar[0][0]
        recommender.compare_streamers(test_streamer, most_similar)


def demo_user_recommendations(recommender):
    """Demo: Recommend streamers based on user viewing history."""
    print("\n" + "="*70)
    print("DEMO 2: USER-BASED RECOMMENDATIONS")
    print("="*70)

    # Simulate a user who has watched a few streamers
    user_history = [
        recommender.idx_to_streamer[i]
        for i in range(min(5, len(recommender.idx_to_streamer)))
    ]

    print(f"\nUser's viewing history:")
    for i, streamer in enumerate(user_history, 1):
        print(f"  {i}. {streamer}")

    print("\nGenerating personalized recommendations...")
    recommendations = recommender.recommend_for_user(user_history, top_k=10)

    print_recommendations(
        recommendations,
        "Personalized Recommendations Based on Viewing History"
    )


def demo_weighted_recommendations(recommender):
    """Demo: Recommendations with weighted user preferences."""
    print("\n" + "="*70)
    print("DEMO 3: WEIGHTED RECOMMENDATIONS")
    print("="*70)

    # Simulate weighted preferences (more watch time = higher weight)
    user_history = [
        (recommender.idx_to_streamer[0], 5.0),  # Watched a lot
        (recommender.idx_to_streamer[1], 3.0),  # Watched medium amount
        (recommender.idx_to_streamer[2], 1.0),  # Watched a little
    ]

    print(f"\nUser's weighted viewing history:")
    for streamer, weight in user_history:
        print(f"  {streamer:<30} (weight: {weight})")

    print("\nGenerating weighted recommendations...")
    recommendations = recommender.recommend_for_user(
        user_history,
        top_k=10,
        aggregation='weighted_avg'
    )

    print_recommendations(
        recommendations,
        "Weighted Recommendations (Favoring Most-Watched Streamers)"
    )


def demo_streamer_profiles(recommender):
    """Demo: View detailed streamer profiles."""
    print("\n" + "="*70)
    print("DEMO 4: STREAMER PROFILES")
    print("="*70)

    # Show profile for a streamer
    test_streamer = recommender.idx_to_streamer[0]
    print(f"\nDetailed profile for: {test_streamer}")
    print("-"*70)

    profile = recommender.get_streamer_profile(test_streamer)

    if profile is not None:
        display_cols = [
            'total_viewership', 'unique_viewers', 'avg_session_duration',
            'num_sessions', 'retention_rate', 'engagement_score',
            'growth_rate', 'peak_time'
        ]

        for col in display_cols:
            if col in profile:
                value = profile[col]
                print(f"{col:<25}: {value:.2f}")


def interactive_mode(recommender):
    """Interactive mode for user queries."""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nCommands:")
    print("  1. similar <streamer_name>  - Find similar streamers")
    print("  2. profile <streamer_name>  - View streamer profile")
    print("  3. compare <name1> <name2>  - Compare two streamers")
    print("  4. list                     - List all streamers")
    print("  5. quit                     - Exit interactive mode")

    while True:
        print("\n" + "-"*70)
        command = input("Enter command: ").strip()

        if not command:
            continue

        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == 'quit' or cmd == 'exit':
            print("Exiting interactive mode...")
            break

        elif cmd == 'similar' and len(parts) > 1:
            streamer_name = parts[1].strip()
            similar = recommender.get_similar_streamers(streamer_name, top_k=10)
            print_recommendations(similar, f"Streamers Similar to '{streamer_name}'")

        elif cmd == 'profile' and len(parts) > 1:
            streamer_name = parts[1].strip()
            profile = recommender.get_streamer_profile(streamer_name)
            if profile is not None:
                print(f"\nProfile for: {streamer_name}")
                print("-"*70)
                display_cols = [
                    'total_viewership', 'unique_viewers', 'avg_session_duration',
                    'retention_rate', 'engagement_score', 'growth_rate'
                ]
                for col in display_cols:
                    if col in profile:
                        print(f"{col:<25}: {profile[col]:.2f}")

        elif cmd == 'compare' and len(parts) > 1:
            names = parts[1].split()
            if len(names) >= 2:
                recommender.compare_streamers(names[0], names[1])
            else:
                print("Usage: compare <streamer1> <streamer2>")

        elif cmd == 'list':
            print("\nAvailable streamers (showing first 50):")
            for i, streamer in enumerate(list(recommender.idx_to_streamer.values())[:50], 1):
                print(f"  {i}. {streamer}")
            total = len(recommender.idx_to_streamer)
            if total > 50:
                print(f"  ... and {total - 50} more")

        else:
            print("Unknown command. Type 'help' for available commands.")


def main():
    """Run the demo."""
    print("\n" + "="*70)
    print("CONTENT-BASED FILTERING DEMO - TWITCH STREAMER RECOMMENDATIONS")
    print("="*70)

    # Check if model exists
    model_path = 'contentBased/models/content_model.pkl'

    if not os.path.exists(model_path):
        print("\nError: Model not found!")
        print("Please run the following scripts first:")
        print("  1. python contentBased/feature_extraction.py")
        print("  2. python contentBased/content_based_model.py")
        sys.exit(1)

    # Load the recommender
    print("\nLoading content-based recommendation model...")
    recommender = ContentBasedRecommender()
    recommender.load_model(model_path)

    # Run demos
    demo_similar_streamers(recommender)
    demo_user_recommendations(recommender)
    demo_weighted_recommendations(recommender)
    demo_streamer_profiles(recommender)

    # Interactive mode
    print("\n" + "="*70)
    response = input("\nEnter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_mode(recommender)

    print("\n✓ Demo complete!")


if __name__ == '__main__':
    main()
