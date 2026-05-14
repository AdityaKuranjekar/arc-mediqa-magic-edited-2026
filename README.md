# 🏥 Medical Vision-Language Agentic RAG — 2026 Edition

### *Efficient Clinical Reasoning Under Constraint: Fast Triage and Asymmetric Partitioning*

> **Authors:** Aditya Kuranjekar · G Prasannapriyan · Prof. Rajiv Misra · Rishu Sharma · Shreya
> **Institution:** Indian Institute of Technology Patna, Bihar, India
> **Dataset:** DermaVQA-DAS (ImageCLEF 2025)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Two Core Innovations](#-the-two-core-innovations-the-why)
- [The 8-Stage Agentic Pipeline](#-the-8-stage-agentic-pipeline-the-how)
- [Results & Performance](#-results--performance)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Step-by-Step Usage](#-step-by-step-usage)
- [Supported Models](#-supported-models)
- [API Requirements & Rate Limits](#-api-requirements--rate-limits)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🔭 Overview

This repository is an **architectural evolution** of the MEDIQA-MAGIC 2025 pipeline originally presented in [*Architecting Clinical Collaboration: Multi-Agent Reasoning Systems for Multimodal Medical VQA*](https://arxiv.org/abs/2507.05520) (Thakrar et al., Georgia Tech, 2025).

The 2026 edition, developed at **IIT Patna**, fundamentally re-architects the system to resolve two critical structural flaws identified in prior multi-agent medical VQA systems. Rather than relying on brute-force ensemble activation or generic multimodal context distribution, this pipeline introduces **confidence-gated triage** and **asymmetric modality siloing** — two targeted modifications that improve both compute efficiency and diagnostic robustness simultaneously.

The system operates on the **DermaVQA-DAS** dataset: 300 patient encounters, 27 questions across 9 clinical families, evaluated with Jaccard-based partial-credit accuracy. It combines local Small Language Models (Llama-3.2-1B, Phi-4-Vision 4B) fine-tuned via LoRA/QLoRA with Gemini 2.5 Flash as the reasoning orchestrator.

**Key result: 78.56% average test accuracy, up from a 38.04% baseline — a +40.52 point absolute improvement — while reducing mean inference latency by 27.4%.**

---

## 💡 The Two Core Innovations (The "Why")

Previous multi-agent RAG systems for medical VQA suffer from two structural flaws that are independent of model quality. This work addresses both with targeted architectural changes requiring no additional training overhead.

---

### Gap 1: The Unconditional Ensemble Problem 🐌

**The Flaw:** State-of-the-art agentic RAG systems unconditionally trigger the full multi-model ensemble, hybrid retrieval engine, and multi-stage synthesis chain for **every single patient query** — regardless of case complexity. A straightforward dermatological question that any single model resolves at 97% confidence still activates the complete 8-agent pipeline, incurring the same latency and compute cost as the most ambiguous edge case.

This is a fundamental mismatch between computational expenditure and diagnostic difficulty — one that creates latency barriers that are particularly damaging in telemedicine settings where real-time response affects clinical outcomes.

**Our Solution: Fast Triage Gatekeeper** ⚡

A lightweight gatekeeper agent sits at the pipeline entrance and processes all three inputs (patient description, images, and clinical query) before any heavy computation is triggered. If the agent's confidence exceeds **95%**, the case exits directly to Final Diagnosis via a bypass path, skipping Phases 2–4 entirely. Approximately **30% of real telemedicine queries** qualify for this bypass — converting a fixed compute cost into a conditional one.

```
Patient Inputs → [Fast Triage Agent]
                      │
           ┌──────────┴──────────┐
     conf > 95%             conf < 95%
           │                     │
    ┌──────▼──────┐    ┌─────────▼──────────┐
    │Final Diagnosis│   │ Full 8-Stage Pipeline│
    └─────────────┘    └────────────────────┘
```

---

### Gap 2: The Consensus Collapse Problem 🧠

**The Flaw:** Existing architectures feed **identical multimodal context** — both patient text and clinical images — to all agents simultaneously. When the image presents a dominant visual cue, all agents anchor on that signal and produce highly correlated outputs. Subtle textual evidence — a patient's occupational exposure history, treatment resistance, or symptom progression description — is systematically underweighted.

This produces *apparent consensus without genuine multi-modal reasoning* — precisely the failure mode that ensemble architectures are designed to prevent.

**Our Solution: Asymmetric Partitioning** 🔀

We enforce strict **modality siloing** at the input level. Patient text is routed **exclusively** to a clinical language model (Llama-3.2-1B) that operates *blind to image data*. Patient images are routed **exclusively** to a vision specialist (Phi-4-Vision 4B) that operates *blind to text*. Their independent reports are then merged by a dedicated **Asymmetric Synthesizer** that preserves contradictions as reduced confidence rather than smoothing them into a softened prediction.

This converts *hidden uncertainty into explicit, auditable uncertainty* — a critical property for clinical adoption, where transparency directly determines physician trust.

```
Patient Text  ──────────► [Clinical Context Expert]  (TEXT ONLY · blind to images)
                                     │
                                     ▼
                          [Diagnosis Extractor] ◄── Clinical Query
                                     ▲
Patient Images ─────────► [Image Analysis Expert]   (IMAGE ONLY · blind to text)
```

---

## ⚙️ The 8-Stage Agentic Pipeline (The "How")

The complete upgraded pipeline flows through four phases containing eight distinct agents:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — GATEKEEPER                                           │
│  ┌───────────────────────────────────┐                          │
│  │  Patient Description              │                          │
│  │  Patient Images          ──────►  Fast Triage Agent          │
│  │  Clinical Query                   │  (conf > 95% → bypass)  │
│  └───────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
         │ conf < 95%
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — ASYMMETRIC PARTITIONING (siloed inputs)              │
│                                                                  │
│  [Text] ──► Clinical Context Expert (Llama-3.2-1B)             │
│                         │                                        │
│                         └────────► Diagnosis Extractor ◄───────┤
│                         ┌────────►                              │
│  [Images] ► Image Analysis Expert (Phi-4-Vision 4B)            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3 — HYBRID KNOWLEDGE RAG                                  │
│                                                                  │
│  Knowledge Retrieval Agent                                       │
│     ├── BM25 Sparse Retrieval                                   │
│     ├── BioBERT Dense Embeddings                                │
│     └── Cross-Encoder Reranking                                 │
│  MedCPT Domain Index (UMLS · PubMed · Clinical Guidelines)     │
│  Evidence Integration Agent                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4 — DECISION SYNTHESIS & SAFETY LOOP                     │
│                                                                  │
│  Asymmetric Synthesizer ──► Final Diagnosis                     │
│       │ conf < 0.75                                             │
│       └──► Self-Reflection Agent ──► Re-Analysis Agent ─────►  │
└─────────────────────────────────────────────────────────────────┘
```

| Stage | Agent | Role |
|-------|-------|------|
| 1 | **Fast Triage Agent** | Confidence-gated bypass for trivial cases (>95% threshold) |
| 2 | **Clinical Context Expert** (Llama-3.2-1B) | Text-only analysis across 13 clinical categories; blind to images |
| 3 | **Image Analysis Expert** (Phi-4-Vision 4B) | Image-only analysis across 10 visual dimensions; blind to text |
| 4 | **Diagnosis Extractor Agent** | Merges independent reports into ranked diagnostic hypotheses |
| 5 | **Knowledge Retrieval Agent** | Hybrid BM25 + BioBERT + Cross-Encoder retrieval over MedCPT index |
| 6 | **Evidence Integration Agent** | Consolidates visual, textual, and retrieved knowledge with adaptive weights |
| 7 | **Asymmetric Synthesizer** | Core reasoning engine; surfaces inter-modal contradictions as reduced confidence |
| 8 | **Safety Loop** | Self-Reflection → Re-Analysis triggered when synthesizer confidence < 0.75 |

---

## 📊 Results & Performance

### Accuracy: Baseline vs. Upgraded Architecture

| Question Type (QID) | Baseline Qwen2-VL | Original RAG | **Upgraded RAG** | Δ vs Original |
|---------------------|:-----------------:|:------------:|:----------------:|:-------------:|
| Body Coverage (CQID010) | 0.31 | 0.47 | **0.65** | +0.18 |
| Anatomical Location (CQID011) | 0.38 | 0.86 | **0.88** | +0.02 |
| Lesion Size (CQID012) | 0.53 | 0.69 | **0.79** | +0.10 |
| Onset Timing (CQID015) | 0.31 | 0.85 | **0.90** | +0.05 |
| Skin Description (CQID020) | 0.31 | 0.56 | **0.65** | +0.09 |
| Itching (CQID025) | 0.42 | 0.84 | **0.95** | +0.11 |
| Lesion Color (CQID034) | 0.01 | 0.51 | **0.55** | +0.04 |
| Lesion Count (CQID035) | 0.72 | 0.82 | **0.95** | +0.13 |
| Texture (CQID036) | 0.37 | 0.64 | **0.75** | +0.11 |
| **Average** | **0.3804** | 0.69 | **0.7856** | **+0.0956** |

> The largest gains appear on **ambiguous multi-modal questions** (Body Coverage, Lesion Count, Itching, Texture) where cross-modal anchor bias previously suppressed textual evidence — confirming that Asymmetric Partitioning directly addresses the consensus collapse failure mode.

### Inference Latency by Triage Path

| Case Type | Original Pipeline | Bypass Path | Full Pipeline |
|-----------|:-----------------:|:-----------:|:-------------:|
| High-confidence (>95%) | ~4,200 ms | **~240 ms** | — |
| Moderate-confidence | ~4,200 ms | — | ~3,950 ms |
| Low-confidence (<0.75, safety loop) | ~4,200 ms | — | ~5,400 ms |
| **Weighted Average** | **~4,200 ms** | — | **~3,050 ms** |

> **27.4% latency reduction** overall. Approximately 30% of validation cases qualified for the >95% bypass path, converting the ensemble compute cost from fixed to conditional.

### Ablation Study Highlights

- Removing the **self-reflection safety loop** degrades average accuracy by ~9–10 percentage points
- Reverting to **shared-context agents** (removing Asymmetric Partitioning) reduces accuracy on multi-label subjective questions by 4–6 points, with the largest drops on Lesion Color and Skin Description
- Confidence thresholds below 0.70 or above 0.85 both underperform the optimal 0.75 threshold

---

## 🏗️ Repository Structure

```
arc-mediqa-magic-2025/
├── 📁 finetuning_pipeline/
│   ├── pipeline.py                          # Complete training + inference pipeline
│   ├── finetuning_pipeline_example_usage.py
│   └── __init__.py
├── 📁 reasoning_pipeline/
│   ├── reasoning_pipeline.py                # Gemini-based 3-stage reasoning system
│   ├── reasoning_pipeline_example_usage.py
│   └── __init__.py
├── 📁 rag_pipeline/
│   ├── rag_pipeline.py                      # 8-stage agentic RAG (2026 upgraded)
│   ├── rag_pipeline_example_usage.py
│   └── __init__.py
├── 📁 evaluation/
│   ├── run_cvqa_eval.py
│   ├── run_segandcvqa_scoring.py
│   └── score_cvqa.py
├── 📁 2025_dataset/
│   ├── train/  (train.json, images_train/)
│   ├── valid/  (valid.json, images_valid/)
│   └── test/   (test.json, images_test/)
├── data_preprocessor.py
├── evaluation_script.py
├── submission_utility.py
├── step1_prepare_data.py
├── step2_smoke_test.py
├── step3_train.py
├── step4_validate.py
├── step5_test_inference.py
├── step6_rag_sanity_test.py
├── step7_rag_full_run.py
├── step8_evaluate.py
├── requirements.txt
└── .env
```

---

## 🚀 Installation & Setup

### Prerequisites

- **OS:** Windows 10/11 or Linux (Ubuntu 22.04+)
- **Python:** 3.10+
- **GPU:** NVIDIA with 8 GB+ VRAM (RTX 3080 / A100 recommended for training; RTX 4060 8 GB sufficient for inference with quantization)
- **CUDA:** 12.4+ with compatible `bitsandbytes`
- **Storage:** 50 GB+ free space

### 1. Clone the Repository

```bash
git clone https://github.com/karishmathakrar/arc-mediqa-magic-2025.git
cd arc-mediqa-magic-2025
```

### 2. Create a Virtual Environment

```bash
python3.10 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If you encounter `UnicodeDecodeError: 'charmap' codec` when reading dataset JSON files, this is a Windows encoding issue. All `open()` calls in `data_preprocessor.py` include `encoding='utf-8'` to handle this. If you see this error, ensure you are running the latest version of `data_preprocessor.py` from this repository.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Hugging Face token (required for gated models like Llama)
HF_TOKEN=your_huggingface_token_here

# Google Gemini API key (required for reasoning and RAG stages)
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: custom HuggingFace model cache directory
HF_HOME=/path/to/huggingface/cache
```

### 5. Place the Dataset

Download the DermaVQA-DAS dataset from [ImageCLEF 2025](https://www.imageclef.org/2025) and place it at:

```
2025_dataset/
├── train/
│   ├── train.json
│   ├── train_cvqa.json
│   ├── option_maps.json
│   ├── closedquestions_definitions_imageclef2025.json
│   └── images_train/
├── valid/
│   ├── valid.json
│   ├── valid_cvqa.json
│   └── images_valid/
└── test/
    ├── test.json
    └── images_test/
```

---

## 📖 Step-by-Step Usage

The pipeline follows a sequential 8-step workflow. Run each step in order.

### Step 1 — Data Preparation

```bash
python step1_prepare_data.py
```

Preprocesses and validates dataset files, generates batch `.pkl` files for training.

### Step 2 — Smoke Test

```bash
python step2_smoke_test.py
```

Runs a 3-sample sanity check to confirm data loading, image paths, and GPU availability are all working before committing to full training.

### Step 3 — Fine-Tune the Vision-Language Model

```python
# Configure in step3_train.py before running:
MODEL_NAME = "Qwen2-VL-2B-Instruct"   # see Supported Models below
USE_COMBINED_DATASET = False            # True to merge train + val
TEST_MODE = False                       # True for a quick 1-epoch test run
```

```bash
python step3_train.py
```

Runs LoRA fine-tuning (rank 8, alpha 16, dropout 0.05) with 4-bit NF4 quantization. On an RTX 4060 8 GB, `Qwen2-VL-2B-Instruct` trains in approximately 2–3 hours per epoch.

### Step 4 — Validation Inference

```bash
python step4_validate.py
```

Runs inference on the validation set and produces aggregated prediction CSV files in `outputs/`. **This step must complete before Step 6.**

### Step 5 — Test Set Inference

```bash
python step5_test_inference.py
```

Runs inference on the test set. Note: test set ground truth is hidden; accuracy cannot be computed locally (see Troubleshooting).

### Step 6 — RAG Sanity Test (1–3 Encounters)

```python
# Configure in step6_rag_sanity_test.py:
GEMINI_MODEL = "gemini-2.5-flash-lite"  # recommended for free tier
NUM_SAMPLES = 1                          # start with 1, increase to 3 when stable
USE_TEST_DATASET = False
USE_FINETUNING = True
```

```bash
python step6_rag_sanity_test.py
```

Runs the full 8-stage agentic pipeline on a small sample to confirm Gemini API connectivity, rate-limit throttling, and knowledge base initialization.

### Step 7 — Full RAG Run

```bash
python step7_rag_full_run.py
```

Processes all encounters with the complete pipeline. Saves intermediate results every N encounters (configurable) to protect against interruptions.

### Step 8 — Evaluation

```bash
python step8_evaluate.py
```

Computes Jaccard-based partial-credit accuracy across all question types and generates the final submission file.

---

### Programmatic Usage

```python
# Fine-tuning pipeline
from finetuning_pipeline.pipeline import FineTuningPipeline

pipeline = FineTuningPipeline(
    model_name="Qwen2-VL-2B-Instruct",
    base_dir="./",
    output_dir="./outputs",
    validate_paths=True,
    setup_environment=True,
)
train_df, val_df = pipeline.prepare_data(use_combined=False, test_mode=False)
trainer = pipeline.train(use_combined=False, test_mode=False)
predictions_df, aggregated_df, formatted = pipeline.run_inference()
```

```python
# 8-Stage Agentic RAG pipeline
from rag_pipeline.rag_pipeline import RAGConfig, RAGPipeline

config = RAGConfig(
    model_name="Qwen2-VL-2B-Instruct",
    use_finetuning=True,
    use_test_dataset=False,
    gemini_model="gemini-2.5-flash-lite",
    max_reflection_cycles=2,
    confidence_threshold=0.75,
    fast_triage_confidence_threshold=0.95,   # Gap 1: bypass threshold
    enforce_modality_separation=True,         # Gap 2: asymmetric partitioning
    base_dir="./",
    output_dir="./outputs",
)

pipeline = RAGPipeline(config)
results = pipeline.process_sample_encounters(num_samples=3)
```

---

## 🤖 Supported Models

| Model | HuggingFace ID | VRAM Required | Notes |
|-------|---------------|:-------------:|-------|
| `Qwen2-VL-2B-Instruct` | `Qwen/Qwen2-VL-2B-Instruct` | ~4 GB | ✅ Default · best for RTX 4060 |
| `Qwen2.5-VL-3B-Instruct` | `Qwen/Qwen2.5-VL-3B-Instruct` | ~6 GB | Good quality/speed balance |
| `Qwen2-VL-7B-Instruct` | `Qwen/Qwen2-VL-7B-Instruct` | ~14 GB | |
| `Qwen2.5-VL-7B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` | ~14 GB | Highest baseline accuracy |
| `gemma-3-4b-it` | `google/gemma-3-4b-it` | ~8 GB | |
| `gemma-3-12b-it` | `google/gemma-3-12b-it` | ~24 GB | |
| `llama-3.2-11b-vision` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | ~22 GB | Requires `HF_TOKEN` |

All models are loaded with **4-bit NF4 quantization** by default to minimize VRAM usage. Flash Attention 2 is enabled where supported.

```python
# Check available models programmatically
from finetuning_pipeline.pipeline import FineTuningPipeline
pipeline = FineTuningPipeline()
print(pipeline.get_available_models())
```

---

## 🔑 API Requirements & Rate Limits

The reasoning and RAG stages (Steps 6–7) require a **Google Gemini API key** set as `GOOGLE_API_KEY` in your `.env` file.

### Recommended Models by Use Case

| Gemini Model | Free Tier RPM | Free Tier RPD | Recommendation |
|---|:---:|:---:|---|
| `gemini-2.5-flash-lite` | 15 | 1,000 | ✅ **Best for free-tier agentic loops** |
| `gemini-2.5-flash` | 10 | 500 | Good for higher quality, halved daily quota |
| `gemini-2.5-pro` | 5 | 100 | Best reasoning, very tight daily cap |
| `gemini-2.0-flash` | 5 | 500 | ⚠️ Deprecated March 2026 — do not use |

> **Important:** The 8-stage pipeline fires approximately 8 Gemini API calls per encounter. At `gemini-2.5-flash-lite`'s 15 RPM with the built-in 10-second mandatory pre-call throttle, a single encounter takes roughly 80–120 seconds. For large datasets (200+ encounters), **upgrading to a paid Gemini API tier is strongly recommended** to avoid hitting daily request limits mid-run.

The pipeline implements automatic rate-limit handling:
- **10-second mandatory sleep** before every API call (caps throughput at ≤6 RPM, safely inside all free-tier windows)
- **Exponential backoff** on 429 errors: 30 s → 60 s → 120 s → 240 s → 480 s (5 retries max)

---

## 🚨 Troubleshooting

### ❌ `division by zero` or "No evaluation files found" in Step 8

**Cause:** You ran inference on the **test set** (`USE_TEST_DATASET = True`). The test set has hidden ground truth — accuracy cannot be computed locally.

**Fix:** Set `USE_TEST_DATASET = False` in `step5_test_inference.py` and `step6_rag_sanity_test.py` to run on the **validation set** when calculating local accuracy metrics. Only switch to `True` when generating your final competition submission.

---

### ❌ `NameError: name 'some_file_path' is not defined` in `data_preprocessor.py`

**Cause:** A manual find-replace accidentally overwrote a variable name with a placeholder string when adding `encoding='utf-8'` to `open()` calls.

**Fix:** On line 120 of `data_preprocessor.py`, replace:
```python
# WRONG
with open(some_file_path, 'r', encoding='utf-8') as f:

# CORRECT
with open(self.config.QUESTIONS_PATH, 'r', encoding='utf-8') as f:
```

---

### ❌ Step 3 prints "Starting pipeline..." then exits silently with no training progress bar

**Cause (most common):** The processed `.pkl` batch files are empty or missing. This happens when a previous run created the output folders but failed before writing data, or when `TEST_MODE = True` triggers training with `save_steps=50` but only 1 total optimisation step exists — so no checkpoint is ever saved and the script exits cleanly.

**Fix:** Ensure `step1_prepare_data.py` and `step2_smoke_test.py` both completed successfully. Then in `step3_train.py`, confirm at least one `.pkl` file exists in `outputs/processed_train_data-{model}/` before training. In `TEST_MODE`, the pipeline now automatically overrides `save_steps=1` to guarantee a checkpoint is always written.

---

### ❌ `429 RESOURCE_EXHAUSTED` errors from Gemini API

**Cause:** The 8-agent pipeline fires API calls faster than your free-tier quota allows.

**Fix options (in order of preference):**
1. Use `gemini-2.5-flash-lite` (15 RPM, 1,000 RPD) — highest free-tier headroom
2. The built-in 10-second throttle + exponential backoff handles transient spikes automatically
3. For datasets of 200+ encounters, upgrade to a Gemini paid tier (Tier 1 provides 150–300 RPM)

---

### ❌ CUDA Out of Memory during training

**Fix options:**
- Switch to a smaller model (`Qwen2-VL-2B-Instruct` instead of 7B)
- Reduce `per_device_train_batch_size` to 1 (already the default)
- Ensure 4-bit NF4 quantization is enabled (default in this pipeline)
- Enable gradient checkpointing (enabled by default)
- Close other GPU-consuming processes before running

---

### ❌ Model loading fails for `llama-3.2-11b-vision`

**Cause:** This is a gated model requiring explicit HuggingFace access.

**Fix:** Request access at [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) and ensure `HF_TOKEN` is set in your `.env`.

---

### Debug Mode

```bash
# Enable verbose logging for CUDA and transformer issues
export TRANSFORMERS_VERBOSITY=debug
export CUDA_LAUNCH_BLOCKING=1
```

---

## 📁 Output Structure

```
outputs/
├── processed_train_data-{model}-V3/     # Preprocessed training batch files (.pkl)
├── processed_val_data-{model}-V3/       # Preprocessed validation batch files
├── processed_test_data-{model}-V3/      # Preprocessed test batch files
├── finetuned-model/                     # LoRA checkpoints + merged weights
├── val_dataset.csv                      # Validation metadata
├── val_aggregated_predictions_{model}_{timestamp}.csv   # Step 4 output
├── test_aggregated_predictions_{model}_{timestamp}.csv  # Step 5 output
├── rag_sample_results_{timestamp}.json  # Step 6 sanity test output
├── validation_data_cvqa_rag_complete_{timestamp}.json   # Step 7 complete results
└── validation_data_cvqa_rag_formatted_{timestamp}.json  # Step 8 submission format
```

---

## 📜 Citation

If you use this work, please cite both the 2026 IIT Patna paper and the original 2025 Georgia Tech paper that this system builds upon:

**2026 Paper (this repository's architecture):**
```bibtex
@article{kuranjekar2026efficient,
  title   = {Efficient Clinical Reasoning Under Constraint: Fast Triage and
             Asymmetric Partitioning in Medical Vision-Language Agentic RAG},
  author  = {Kuranjekar, Aditya and Prasannapriyan, G and Misra, Rajiv and
             Sharma, Rishu and Shreya},
  institution = {Indian Institute of Technology Patna},
  year    = {2026}
}
```

**2025 Paper (original baseline architecture):**
```bibtex
@article{thakrar2025architecting,
  title   = {Architecting Clinical Collaboration: Multi-Agent Reasoning Systems
             for Multimodal Medical VQA},
  author  = {Thakrar, Karishma and Basavatia, Shreyas and Daftardar, Akshay},
  journal = {arXiv preprint arXiv:2507.05520},
  year    = {2025}
}
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

The authors thank **IIT Patna** for providing GPU compute resources used in the fine-tuning experiments, and the **ImageCLEF 2025** organizers for making the DermaVQA-DAS dataset publicly available. This work builds directly on the foundational architecture of Thakrar et al. (2025) from Georgia Institute of Technology.

---

<div align="center">
  <sub>Built at IIT Patna · 2026 · For questions, open a GitHub Issue</sub>
</div>