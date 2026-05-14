import os
import glob
import datetime

# Configuration
OUTPUT_DIR = r"E:\arc-mediqa-magic-2025-main\arc-mediqa-magic-2025-main\outputs"
TOTAL_ENCOUNTERS = 504
PATTERN = os.path.join(OUTPUT_DIR, "diagnosis_based_rag_results_ENC*.json")

def calculate_eta():
    files = glob.glob(PATTERN)
    if not files:
        print("No individual encounter results found in outputs/.")
        return

    # Sort files by modification time
    files.sort(key=os.path.getmtime)
    
    # We want the most recent ones (from the latest sanity test)
    # The user said they just ran 10 encounters.
    recent_files = files[-10:] if len(files) >= 10 else files
    
    if len(recent_files) < 2:
        print("Not enough files to calculate an average. Please run at least 2 encounters.")
        return

    # Calculate time between first and last of these recent files
    start_time = os.path.getmtime(recent_files[0])
    end_time = os.path.getmtime(recent_files[-1])
    
    total_duration = end_time - start_time
    avg_per_encounter = total_duration / (len(recent_files) - 1)
    
    total_projected_seconds = avg_per_encounter * TOTAL_ENCOUNTERS
    total_projected_hours = total_projected_seconds / 3600
    
    print("=" * 50)
    print("RAG PIPELINE - ETA CALCULATOR")
    print("=" * 50)
    print(f"Sample size evaluated   : {len(recent_files)} encounters")
    print(f"Avg time per encounter  : {avg_per_encounter:.2f} seconds")
    print(f"Total encounters        : {TOTAL_ENCOUNTERS}")
    print("-" * 50)
    print(f"Projected total time    : {total_projected_hours:.2f} hours")
    
    if total_projected_hours < 12:
        print("\n✅ STATUS: On track to finish within the 12-hour deadline.")
    else:
        print(f"\n⚠️ WARNING: Projected time ({total_projected_hours:.2f}h) exceeds the 12-hour deadline.")
        print(f"   Consider reducing dataset size to ~{int((12 * 3600) / avg_per_encounter)} encounters.")
    print("=" * 50)

if __name__ == "__main__":
    calculate_eta()
