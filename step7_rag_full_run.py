#!/usr/bin/env python3
"""
STEP 7 — RAG Pipeline: Full Production Run (All Encounters)
Runs the 8-stage agentic RAG pipeline on the FULL test/validation dataset.
Must be run AFTER step5_test_inference.py (for test) or step4_validate.py (for val).
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 7: RAG Pipeline — Full Production Run")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# ▶  Configure below
# ══════════════════════════════════════════════════════════════

GEMINI_MODEL = "gemini-2.5-flash"

# True  = run on test dataset  (for final submission)
# False = run on validation set (for benchmarking your score)
USE_TEST_DATASET = True

# True = use fine-tuned model predictions from Phase 1
USE_FINETUNING = True

# Save a checkpoint every N encounters (in case of interruption)
SAVE_EVERY_N = 1

# ══════════════════════════════════════════════════════════════

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    print("❌ GOOGLE_API_KEY not set — cannot run RAG pipeline!")
    sys.exit(1)

print(f"\nGemini model     : {GEMINI_MODEL}")
print(f"Dataset          : {'TEST (final submission)' if USE_TEST_DATASET else 'VALIDATION (benchmarking)'}")
print(f"Model predictions: {'fine-tuned' if USE_FINETUNING else 'base'}")
print(f"Checkpoints every: {SAVE_EVERY_N} encounters")
print()

from rag_pipeline import RAGConfig, RAGPipeline

config = RAGConfig(
    use_finetuning=USE_FINETUNING,
    use_test_dataset=USE_TEST_DATASET,
    gemini_model=GEMINI_MODEL,
    max_reflection_cycles=2,
    confidence_threshold=0.75,
    fast_triage_confidence_threshold=0.95,   # Gap 1: Fast Triage Gatekeeper
    enforce_modality_separation=True,          # Gap 2: Asymmetric Partitioning
    save_intermediate_results=True,
    intermediate_save_frequency=SAVE_EVERY_N,
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
)

print("Initializing RAG pipeline...")
pipeline = RAGPipeline(config)

print("\nProcessing ALL encounters (this may take a while)...")
print("Intermediate checkpoints will be saved to outputs/ every")
print(f"{SAVE_EVERY_N} encounters in case of interruption.\n")

complete_results, formatted_predictions = pipeline.process_all_encounters(num_samples=20)

print("\n" + "=" * 60)
print(f"✅ Full RAG pipeline complete!")
print(f"   Encounters processed         : {len(complete_results)}")
print(f"   Formatted prediction entries : {len(formatted_predictions)}")
print()
print("   Final output files in:")
print(f"   {OUTPUT_DIR}")
print()
dataset_label = "test" if USE_TEST_DATASET else "validation"
print("   Files created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "rag" in f.lower() and f.endswith(".json"):
        fsize = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"   ├── {f}  ({fsize:.1f} KB)")
print()
print("   Proceed to Step 8 to evaluate your predictions.")
print("=" * 60)
