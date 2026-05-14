#!/usr/bin/env python3
"""
STEP 3 — Fine-tune the Vision-Language Model (Full Training)
LoRA fine-tunes the selected model on all training data.
This step requires a CUDA GPU.
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 3: Fine-Tuning (Full Training Run)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# ▶  CONFIGURE HERE — change only these three values if needed
# ══════════════════════════════════════════════════════════════

# Model to fine-tune. Choose one:
#   "Qwen2-VL-2B-Instruct"    (recommended, needs ~8 GB VRAM)
#   "Qwen2.5-VL-3B-Instruct"  (better quality, needs ~10 GB VRAM)
#   "Qwen2.5-VL-7B-Instruct"  (best quality,   needs ~20 GB VRAM)
#   "Qwen2-VL-7B-Instruct"    (alternative 7B)
#   "llama-3.2-11b-vision"    (needs HF_TOKEN + ~28 GB VRAM)
#   "gemma-3-4b-it"           (needs ~12 GB VRAM)
MODEL_NAME = "Qwen2-VL-2B-Instruct"

# Set to True to merge train + validation data for training
# (gives more data but you lose the ability to validate separately)
USE_COMBINED_DATASET = False

# Set to True for a quick 1-epoch test run on a tiny subset
# Set to False for real training (3 epochs, all data)
TEST_MODE = False

# ══════════════════════════════════════════════════════════════

import torch
print(f"\nModel          : {MODEL_NAME}")
print(f"Combined data  : {USE_COMBINED_DATASET}")
print(f"Test mode      : {TEST_MODE}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\nStarting pipeline...")

# --- NEW HARDENED IMPORT LOGIC ---
try:
    import importlib.util
    import sys
    
    # Define the exact path to your pipeline file
    pipeline_file = os.path.join(BASE_DIR, "finetuning_pipeline", "pipeline.py")
    print(f"DEBUG: Manually loading from: {pipeline_file}")
    
    if not os.path.exists(pipeline_file):
        print(f"CRITICAL ERROR: File not found at {pipeline_file}")
        sys.exit(1)

    # This bypasses the standard 'import' system and loads the file directly into memory
    spec = importlib.util.spec_from_file_location("pipeline_direct", pipeline_file)
    pipe_module = importlib.util.module_from_spec(spec)
    
    print("DEBUG: Executing module code (This is where silent crashes happen)...")
    # This line executes the 'import' statements inside pipeline.py
    spec.loader.exec_module(pipe_module) 
    
    FineTuningPipeline = pipe_module.FineTuningPipeline
    print("DEBUG: Import successful!")

    # 3. Create instance
    pipeline = FineTuningPipeline(
        model_name=MODEL_NAME,
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        validate_paths=True,
        setup_environment=True,
    )
    print("DEBUG: Pipeline object created successfully.")

except Exception as e:
    print(f"\n[!] CAUGHT ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except BaseException as e:
    # This catches things that 'Exception' misses, like SystemExit or hard binary errors
    print(f"\n[!] CAUGHT LOW-LEVEL ERROR: {e}")
    sys.exit(1)

print("\n[1/2] Preparing training data...")
train_df, val_df = pipeline.prepare_data(
    use_combined=USE_COMBINED_DATASET,
    test_mode=TEST_MODE,
)
print(f"      Train rows: {len(train_df)}")
print(f"      Val rows  : {len(val_df)}")

print("\n[2/2] Starting LoRA fine-tuning...")
print("      Checkpoints will be saved to:")
print(f"      {pipeline.get_config().MODEL_SAVE_DIRECTORY}")
print("      Monitor training with:")
print(f"      tensorboard --logdir {OUTPUT_DIR}\\finetuned-model")
print()

# CHANGE: Pass the dataframes into the train method
trainer = pipeline.train(
    use_combined=USE_COMBINED_DATASET,
    test_mode=TEST_MODE,
    train_df=train_df,  # Added this line
    val_df=val_df       # Added this line
)

print("\n" + "=" * 60)
if trainer is not None:
    print(f"✅ Training complete!")
    print(f"   Checkpoint directory: {pipeline.get_config().MODEL_SAVE_DIRECTORY}")
    print("   Proceed to Step 4 to run validation inference.")
else:
    print("❌ Training failed — check the error messages above.")
print("=" * 60)
