#!/usr/bin/env python3
"""
STEP 6 — RAG Pipeline: Sanity Test on 1 Encounter
Runs the full 8-stage agentic RAG pipeline on 1 random encounter
BEFORE committing to the full dataset.
Must be run AFTER step4_validate.py (needs the aggregated CSV files).

Free-tier model guide (post-Dec 2025 quota cuts)
-------------------------------------------------
Model                    RPM   RPD    Notes
gemini-2.5-flash-lite     15  1000   ← RECOMMENDED for agentic loops
gemini-2.5-flash          10   500   good balance of quality & quota
gemini-2.5-pro             5   100   best reasoning, very tight daily cap
gemini-2.0-flash           5   500   DEPRECATED — retired Mar 3 2026
gemini-2.0-flash-exp       2    --   experimental, lowest limits

For an 8-agent pipeline each encounter fires ~8 Gemini calls.
  - gemini-2.5-flash-lite @ 15 RPM → comfortable headroom
  - gemini-2.5-flash      @ 10 RPM → just fits with the 10 s throttle
  - gemini-2.0-flash      @  5 RPM → will still 429 even with throttling
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 6: RAG Pipeline — Sanity Test (1 Encounter)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# ▶  Configure below if needed
# ══════════════════════════════════════════════════════════════

# Paid-tier configuration
# gemini-1.5-flash      : High speed, excellent vision
# gemini-1.5-pro        : Maximum reasoning, slower
GEMINI_MODEL = "gemini-2.5-flash"

# High-speed sanity check for presentation
NUM_SAMPLES = 10

# Set to False to run on validation set; True to run on test set.
USE_TEST_DATASET = False

# Use predictions from the fine-tuned model (True) or base model (False).
USE_FINETUNING = True

# ══════════════════════════════════════════════════════════════

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

import google.genai as genai_check
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    print("❌ GOOGLE_API_KEY not set in .env — cannot run RAG pipeline!")
    print("   Add: GOOGLE_API_KEY=your_key_here  to your .env file")
    sys.exit(1)
else:
    print(f"   ✅ GOOGLE_API_KEY is set")

print(f"\nGemini model     : {GEMINI_MODEL}")
print(f"Dataset          : {'test' if USE_TEST_DATASET else 'validation'}")
print(f"Model predictions: {'fine-tuned' if USE_FINETUNING else 'base'}")
print(f"Sample encounters: {NUM_SAMPLES}")
print(f"Throttle         : 10 s mandatory sleep before every Gemini call")
print(f"Max retries      : 5 (backoff 30 s → 60 s → 120 s → 240 s → 480 s)")

from rag_pipeline import RAGConfig, RAGPipeline

config = RAGConfig(
    model_name="Qwen2-VL-2B-Instruct",
    use_finetuning=USE_FINETUNING,
    use_test_dataset=USE_TEST_DATASET,
    gemini_model=GEMINI_MODEL,
    max_reflection_cycles=2,
    confidence_threshold=0.75,
    fast_triage_confidence_threshold=0.95,
    enforce_modality_separation=True,
    save_intermediate_results=True,
    intermediate_save_frequency=3,
    base_dir=BASE_DIR,
    output_dir=OUTPUT_DIR,
)

print("\nInitializing RAG pipeline (knowledge base loads on first run)...")
pipeline = RAGPipeline(config)

print(f"\nProcessing {NUM_SAMPLES} sample encounter(s) at full speed...")
results = pipeline.process_sample_encounters(num_samples=NUM_SAMPLES)

print("\n" + "=" * 60)
print(f"✅ RAG sanity test complete!")
print(f"   Encounters processed: {len(results)}")
print()
print("   Check the output file in:")
print(f"   {OUTPUT_DIR}  (look for rag_sample_results_*.json)")
print()
print("   If results look good:")
print("   • Increase NUM_SAMPLES to 3 and re-run for a fuller sanity check.")
print("   • Then proceed to Step 7 for the full run.")
print("=" * 60)