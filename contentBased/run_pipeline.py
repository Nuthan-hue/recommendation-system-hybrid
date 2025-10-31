"""
Quick-start Pipeline Script
Runs the entire content-based filtering pipeline from start to finish.
"""

import os
import sys
import time


def run_step(step_name, script_path):
    """Run a pipeline step and report status."""
    print("\n" + "="*70)
    print(f"STEP: {step_name}")
    print("="*70)

    start_time = time.time()

    # Import and run the script
    try:
        # Save current directory
        original_dir = os.getcwd()

        # Change to script directory
        script_dir = os.path.dirname(script_path)
        if script_dir:
            os.chdir(script_dir)

        # Import and run main function
        module_name = os.path.basename(script_path).replace('.py', '')
        if module_name == 'feature_extraction':
            from feature_extraction import main
        elif module_name == 'content_based_model':
            from content_based_model import main

        main()

        # Restore directory
        os.chdir(original_dir)

        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.2f} seconds")
        return True

    except Exception as e:
        print(f"\n✗ Error in {step_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_data_exists():
    """Check if the dataset exists."""
    data_path = '../data/100k_a.csv'

    if not os.path.exists(data_path):
        print("\n" + "="*70)
        print("ERROR: Dataset not found!")
        print("="*70)
        print(f"\nExpected location: {data_path}")
        print("\nPlease ensure the Twitch dataset (100k_a.csv) is in the data/ directory")
        return False

    file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
    print(f"\n✓ Dataset found: {data_path} ({file_size:.2f} MB)")
    return True


def main():
    """Run the complete content-based filtering pipeline."""
    print("\n" + "="*70)
    print("CONTENT-BASED FILTERING - AUTOMATED PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Extract streamer features from viewing data")
    print("  2. Train the content-based recommendation model")
    print("  3. Display summary statistics")
    print("\nEstimated time: 3-5 minutes")

    # Check if user wants to proceed
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Pipeline cancelled.")
        return

    start_time = time.time()

    # Check if data exists
    if not check_data_exists():
        sys.exit(1)

    # Step 1: Feature Extraction
    success = run_step(
        "Feature Extraction",
        "feature_extraction.py"
    )
    if not success:
        print("\n✗ Pipeline failed at feature extraction")
        sys.exit(1)

    # Step 2: Model Training
    success = run_step(
        "Model Training",
        "content_based_model.py"
    )
    if not success:
        print("\n✗ Pipeline failed at model training")
        sys.exit(1)

    # Pipeline complete
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {total_time:.2f} seconds")
    print("\nGenerated files:")
    print("  - contentBased/processed/streamer_features.csv")
    print("  - contentBased/processed/streamer_features.pkl")
    print("  - contentBased/models/content_model.pkl")

    print("\nNext steps:")
    print("  - Run 'python demo.py' to try the recommendation system")
    print("  - Import ContentBasedRecommender in your own scripts")
    print("  - Check README.md for API usage examples")

    # Ask if user wants to run demo
    response = input("\nRun demo now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nLaunching demo...")
        try:
            from demo import main as demo_main
            demo_main()
        except Exception as e:
            print(f"Error running demo: {e}")


if __name__ == '__main__':
    main()