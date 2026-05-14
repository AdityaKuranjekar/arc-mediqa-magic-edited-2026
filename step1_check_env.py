#!/usr/bin/env python3
"""
STEP 1 — Environment & Dataset Check
Run this first to verify everything is set up correctly before training.
"""

import os
import sys

# Ensure UTF-8 output for emojis on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("=" * 60)
print("STEP 1: Environment & Dataset Check")
print("=" * 60)

# ── 1. Python version ─────────────────────────────────────────
print(f"\n[1/6] Python: {sys.version.split()[0]}")

# ── 2. PyTorch & CUDA ─────────────────────────────────────────
try:
    import torch
    print(f"[2/6] PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"      GPU {i}: {name}  ({mem:.1f} GB VRAM)")
    else:
        print("      ⚠️  No CUDA GPU detected — training will be very slow on CPU!")
except ImportError:
    print("[2/6] ❌ PyTorch not installed — run: pip install -r requirements.txt")
    sys.exit(1)

# ── 3. Key packages ───────────────────────────────────────────
print("[3/6] Checking key packages...")
packages = {
    "transformers":        "transformers",
    "peft":                "peft",
    "trl":                 "trl",
    "bitsandbytes":        "bitsandbytes",
    "accelerate":          "accelerate",
    "sentence_transformers":"sentence_transformers",
    "lancedb":             "lancedb",
    "google.genai":        "google-genai",
    "dotenv":              "python-dotenv",
}
missing = []
for import_name, pkg_name in packages.items():
    try:
        __import__(import_name)
        print(f"      ✅ {pkg_name}")
    except ImportError:
        print(f"      ❌ {pkg_name}  ← MISSING")
        missing.append(pkg_name)

if missing:
    print(f"\n  Fix: pip install {' '.join(missing)}")

# ── 4. Dataset paths ──────────────────────────────────────────
print("[4/6] Checking dataset files...")
required = {
    "Train JSON":         os.path.join(BASE_DIR, "2025_dataset", "train", "train.json"),
    "Train CVQA":         os.path.join(BASE_DIR, "2025_dataset", "train", "train_cvqa.json"),
    "Train questions":    os.path.join(BASE_DIR, "2025_dataset", "train", "closedquestions_definitions_imageclef2025.json"),
    "Val JSON":           os.path.join(BASE_DIR, "2025_dataset", "valid", "valid.json"),
    "Val CVQA":           os.path.join(BASE_DIR, "2025_dataset", "valid", "valid_cvqa.json"),
    "Test JSON":          os.path.join(BASE_DIR, "2025_dataset", "test", "test.json"),
    "Train images dir":   os.path.join(BASE_DIR, "2025_dataset", "train", "images_train"),
    "Val images dir":     os.path.join(BASE_DIR, "2025_dataset", "valid", "images_valid"),
    "Test images dir":    os.path.join(BASE_DIR, "2025_dataset", "test", "images_test"),
}
all_ok = True
for label, path in required.items():
    exists = os.path.exists(path)
    icon = "✅" if exists else "❌"
    print(f"      {icon} {label}: {path}")
    if not exists:
        all_ok = False

# ── 5. .env / API keys ────────────────────────────────────────
print("[5/6] Checking API keys...")
env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_file):
    print(f"      ✅ .env file found: {env_file}")
    from dotenv import load_dotenv
    load_dotenv(env_file)
else:
    print(f"      ⚠️  .env file NOT found at {env_file}")
    print("         Create it with:")
    print("         HF_TOKEN=hf_your_token_here")
    print("         GOOGLE_API_KEY=your_gemini_key_here")

hf = os.getenv("HF_TOKEN", "")
gk = os.getenv("GOOGLE_API_KEY", "")
print(f"      HF_TOKEN      : {'✅ set' if hf else '⚠️  not set (needed for Llama models)'}")
print(f"      GOOGLE_API_KEY: {'✅ set' if gk else '❌ not set (REQUIRED for RAG Phase 2)'}")

# ── 6. Outputs directory ──────────────────────────────────────
print("[6/6] Creating outputs directory...")
outputs_dir = os.path.join(BASE_DIR, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
print(f"      ✅ {outputs_dir}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok and not missing:
    print("✅  All checks passed — ready to proceed to Step 2!")
else:
    print("⚠️  Some checks failed — fix the issues above before continuing.")
print("=" * 60)
