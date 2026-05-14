#!/usr/bin/env python3
"""
STEP 8 — Evaluate Predictions Against Ground Truth
Scores your predictions on the validation set.
Must be run AFTER step4_validate.py (Phase 1) or step7_rag_full_run.py (Phase 2).
NOTE: Evaluation only works on the VALIDATION set (not the test set, which has no ground truth).
"""

import os
import sys
import glob
import json

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

print("=" * 60)
print("STEP 8: Evaluate Predictions")
print("=" * 60)

REFERENCE_FILE = os.path.join(BASE_DIR, "2025_dataset", "valid", "valid_cvqa.json")

if not os.path.exists(REFERENCE_FILE):
    print(f"❌ Reference file not found: {REFERENCE_FILE}")
    sys.exit(1)

# ─── Find the latest prediction JSON to evaluate ─────────────
# Priority: RAG formatted > Phase 1 formatted
# Both are in outputs/ and follow the pattern *cvqa_*.json

print("\nScanning for prediction files in outputs/...")

# Gather all candidate prediction JSON files
candidates = []

# Phase 2 RAG formatted predictions (preferred)
rag_files = glob.glob(os.path.join(OUTPUT_DIR, "*rag_formatted*.json"))
for f in rag_files:
    if "validation" in f or "valid" in f:
        candidates.append(("RAG-formatted", f))

# Phase 1 val formatted predictions (fallback)
phase1_files = glob.glob(os.path.join(OUTPUT_DIR, "val_data_cvqa_sys_*.json"))
for f in phase1_files:
    candidates.append(("Phase-1-formatted", f))

if not candidates:
    print("❌ No prediction files found in outputs/")
    print("   Run step4_validate.py (Phase 1) or step7_rag_full_run.py (Phase 2) first.")
    sys.exit(1)

# Sort by modification time to get the newest file
candidates.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
source, PREDICTION_FILE = candidates[0]

print(f"\n✅ Selected prediction file ({source}):")
print(f"   {PREDICTION_FILE}")
print(f"\n   Reference file:")
print(f"   {REFERENCE_FILE}")

# Quick sanity check
with open(PREDICTION_FILE) as f:
    preds = json.load(f)
print(f"\n   Prediction entries: {len(preds)}")

EVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "evaluation")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# ─── Run official evaluation ──────────────────────────────────
print("\nRunning evaluation...")
from evaluation_script import evaluate_predictions, print_evaluation_summary

result = evaluate_predictions(REFERENCE_FILE, PREDICTION_FILE, EVAL_OUTPUT_DIR)

if result is None:
    print("❌ Evaluation failed — check that the evaluation/ directory with score_cvqa.py exists.")
    sys.exit(1)

results, results_file = result
print_evaluation_summary(results)

print("\n" + "=" * 60)
print(f"✅ Evaluation complete!")
print(f"   Detailed scores saved to: {results_file}")
print("=" * 60)
