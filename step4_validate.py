#!/usr/bin/env python3
"""
STEP 4 — Validation Inference (Phase 1)
Runs the fine-tuned model on the validation set and saves predictions.
Must be run AFTER step3_train.py.
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 4: Validation Inference (Fine-tuned Model)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# ▶  Must match the MODEL_NAME used in step3_train.py
# ══════════════════════════════════════════════════════════════
MODEL_NAME = "Qwen2-VL-2B-Instruct"

# Set to None to run on all validation samples.
# Set to a number (e.g., 10) to test on a small subset first.
MAX_SAMPLES = None
# ══════════════════════════════════════════════════════════════

print(f"\nModel       : {MODEL_NAME}")
print(f"Max samples : {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
print(f"Output dir  : {OUTPUT_DIR}")

from finetuning_pipeline.pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(
    model_name=MODEL_NAME,
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
    validate_paths=True,
    setup_environment=True,
)

print("\nRunning inference on validation set with fine-tuned model...")
print("Optimizing for RTX 4060 (8GB VRAM): Loading in 4-bit, direct adapter application.\n")

predictions_df, aggregated_df, formatted_preds = pipeline.run_inference(
    use_finetuning=True,     # Use fine-tuned checkpoint
    test_mode=False,          # Run on validation set
    max_samples=MAX_SAMPLES,
    merge_weights=False,      # Set to False to avoid OOM during merge on 8GB cards
    load_in_4bit=True         # Required for 8GB VRAM
)

print("\n" + "=" * 60)
print(f"✅ Validation inference complete!")
print(f"   Individual predictions  : {len(predictions_df)} rows")
print(f"   Aggregated by encounter : {len(aggregated_df)} rows")
print(f"   Eval-ready JSON entries : {len(formatted_preds)}")
print()
print("   Output files saved in:")
print(f"   {OUTPUT_DIR}")
print()
print("   Files created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "val_" in f and any(f.endswith(ext) for ext in [".csv", ".json"]):
        fsize = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"   ├── {f}  ({fsize:.1f} KB)")
print()
print("   Proceed to Step 5 to run on the test set.")
print("=" * 60)
