#!/usr/bin/env python3
"""
STEP 5 — Test Set Inference (Phase 1)
Processes the test images and runs the fine-tuned model on the test set.
Must be run AFTER step3_train.py.
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 5: Test Set Inference (Fine-tuned Model)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# ▶  Must match MODEL_NAME used in step3_train.py
# ══════════════════════════════════════════════════════════════
MODEL_NAME = "Qwen2-VL-2B-Instruct"

# Set to None to run on all test samples.
MAX_SAMPLES = None
# ══════════════════════════════════════════════════════════════

print(f"\nModel       : {MODEL_NAME}")
print(f"Max samples : {MAX_SAMPLES if MAX_SAMPLES else 'All'}")

from finetuning_pipeline.pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(
    model_name=MODEL_NAME,
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
    validate_paths=True,
    setup_environment=True,
)

print("\n[1/2] Pre-processing test images into batch files...")
total_test = pipeline.training_pipeline.process_test_dataset(batch_size=100)
print(f"      ✅ {total_test} test samples processed")

print("\n[2/2] Running inference on test set...")
print("Optimizing for RTX 4060 (8GB VRAM): Loading in 4-bit, direct adapter application.\n")

predictions_df, aggregated_df, formatted_preds = pipeline.run_inference(
    use_finetuning=True,     # Use fine-tuned checkpoint
    test_mode=True,           # Run on TEST set (not validation)
    max_samples=MAX_SAMPLES,
    merge_weights=False,      # Set to False to avoid OOM during merge on 8GB cards
    load_in_4bit=True         # Required for 8GB VRAM
)

print("\n" + "=" * 60)
print(f"✅ Test inference complete!")
print(f"   Individual predictions  : {len(predictions_df)} rows")
print(f"   Aggregated by encounter : {len(aggregated_df)} rows")
print(f"   Eval-ready JSON entries : {len(formatted_preds)}")
print()
print("   Output files saved in:")
print(f"   {OUTPUT_DIR}")
print()
print("   Files created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "test_" in f and any(f.endswith(ext) for ext in [".csv", ".json"]):
        fsize = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"   ├── {f}  ({fsize:.1f} KB)")
print()
print("   Proceed to Step 6 to run the RAG agentic pipeline.")
print("=" * 60)
