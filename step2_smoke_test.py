#!/usr/bin/env python3
"""
STEP 2 — Smoke Test (Data Loading Only, No GPU Required)
Verifies that data can be loaded and preprocessed correctly before committing to full training.
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 2: Smoke Test (Data Loading Check)")
print("=" * 60)

# ── Model selection ───────────────────────────────────────────
# Change this to the model you want to fine-tune.
# Options:
#   "Qwen2-VL-2B-Instruct"    ← recommended for <16 GB VRAM
#   "Qwen2.5-VL-3B-Instruct"  ← good balance
#   "Qwen2.5-VL-7B-Instruct"  ← best quality (needs ~20 GB VRAM)
#   "llama-3.2-11b-vision"    ← needs HF_TOKEN + ~28 GB VRAM
MODEL_NAME = "Qwen2-VL-2B-Instruct"

print(f"\nSelected model : {MODEL_NAME}")
print(f"Base directory : {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

from finetuning_pipeline.pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(
    model_name=MODEL_NAME,
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
    validate_paths=True,
    setup_environment=True,
)

print("\n[1/2] Preparing data in test mode (small subset)...")
train_df, val_df = pipeline.prepare_data(
    use_combined=False,
    test_mode=True,
    min_data_size=5,
)
print(f"      ✅ Train rows loaded : {len(train_df)}")
print(f"      ✅ Val rows loaded   : {len(val_df)}")

print("\n[2/2] Inspecting a sample of processed training data...")
pipeline.training_pipeline.inspect_processed_data(
    num_samples=2,
    data_type="train",
)

print("\n" + "=" * 60)
print("✅ Smoke test passed — data pipeline is working correctly.")
print("   Proceed to Step 3 to start full training.")
print("=" * 60)
