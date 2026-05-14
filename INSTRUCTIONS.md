# 🚀 Complete End-to-End Training Guide: ARC-MEDIQA Magic 2025

This guide covers absolutely everything required to go from a fresh machine to a fully trained, Agentic RAG-powered medical vision-language model, specifically tailored for Windows and an NVIDIA RTX 4060 GPU (8GB VRAM / 32GB RAM).

---

## 🔑 PHASE 1: Credentials & API Keys

The pipeline requires two specific keys to function: HuggingFace (to download the AI models) and Google Gemini (to power the Agentic Reasoning loops).

### 1. Get your HuggingFace Token
1. Go to **[HuggingFace](https://huggingface.co/)** and create an account or log in.
2. Click your profile picture in the top-right corner and select **Settings**.
3. On the left sidebar, click **Access Tokens**.
4. Click the **"Create new token"** button.
5. Name it `arc-mediqa-2025`.
6. Under **Token Type**, select **Read** (you only need read access to download models).
7. Click **Generate Token** and copy the string that starts with `hf_...`.

### 2. Get your Google Gemini API Key
1. Go to **[Google AI Studio](https://aistudio.google.com/app/apikey)** and log in with your Google account.
2. Click the **Create API key** button.
3. Select an existing Google Cloud project or let it create one for you automatically.
4. Copy the generated key string (it usually starts with `AIzaSy...`).

### 3. Configure the `.env` File
1. Open your project folder in File Explorer.
2. You will see a file named `.env.template` (or just `.env` if you've already renamed it).
3. Open this file in your code editor.
4. Paste your keys so the file looks exactly like this:
   ```env
   HF_TOKEN=hf_YourActualHuggingFaceTokenHere
   GOOGLE_API_KEY=AIzaSyYourActualGeminiKeyHere
   ```
5. Save the file. If it is named `.env.template`, rename it to just `.env`.

---

## 🖥️ PHASE 2: GPU Environment Setup

Because you are using an NVIDIA GPU, you MUST use PyTorch with CUDA support. The CPU version currently installed will take weeks to train.

### 1. Verify NVIDIA Drivers
1. Open PowerShell and run:
   ```powershell
   & "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
   ```
2. **If it works:** Look at the top right of the table for **CUDA Version** (e.g., 12.4).
3. **If it fails:** Go to the [NVIDIA Drivers Download Page](https://www.nvidia.com/Download/index.aspx), download the driver for your GPU, install it using "Express Install", and restart your computer.

### 2. Uninstall CPU PyTorch
In PowerShell, run this exact command to remove the slow CPU version:
```powershell
python -m pip uninstall torch torchvision torchaudio -y
```

### 3. Install CUDA PyTorch (For CUDA 12.4)
Run this command to download the GPU-accelerated version (this will take about 10 minutes as it is ~2.5GB):
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Verify GPU Detection
Run this to confirm PyTorch now sees your GPU:
```powershell
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"
```
*(It must print `CUDA Ready: True` before you continue).*

---

## 📂 PHASE 3: Dataset Acquisition

The code is ready, but the actual medical images are missing because they are private competition data.

### 1. Register and Download
1. Go to the **[ImageCLEF 2025 MEDIQA-MAGIC page](https://www.imageclef.org/2025/medical/mediqa)**.
2. Register for the competition and accept the data usage agreement.
3. Download the 3 zipped image files for Train, Validation, and Test datasets.

### 2. Create the Folder Structure
Open PowerShell and create the required directories:
```powershell
mkdir 2025_dataset\train\images_train
mkdir 2025_dataset\valid\images_valid
mkdir 2025_dataset\test\images_test
```

### 3. Extract the Images
Unzip the downloaded files and place the actual `.jpg` / `.png` images directly into their respective folders created above.

---

## 🛠️ PHASE 4: Pipeline Execution (Step-by-Step)

Now that your keys, GPU, and datasets are perfect, we run the automated pipeline scripts in numerical order.

### Step 1: Final Environment Check
```powershell
python step1_check_env.py
```
> **Expected Result:** Everything should say ✅. There should be no warnings about CPU or missing image directories.

### Step 2: Smoke Test
```powershell
python step2_smoke_test.py
```
> **Expected Result:** The system will briefly load the Qwen model into your GPU VRAM, process one image, and exit. This proves you won't crash due to Out-Of-Memory errors during actual training.

### Step 3: Fine-Tuning the Model
1. Open `step3_train.py` in your code editor.
2. Ensure lines 30-38 look like this (to ensure it fits on your GPU):
   ```python
   MODEL_NAME = "Qwen2-VL-2B-Instruct" 
   USE_COMBINED_DATASET = False         
   TEST_MODE = False                    
   ```
3. Run the training script:
   ```powershell
   python step3_train.py
   ```
> **What Happens:** The script processes the images, loads the model in 4-bit mode, and trains Low-Rank Adapters (LoRA) specifically on the dermatology questions. This will take several hours. Let it run uninterrupted.

### Step 4: Validation
Once training is 100% complete, run:
```powershell
python step4_validate.py
```
> **What Happens:** The model uses its newly learned knowledge to predict answers for the Validation dataset.

### Step 5: Test Basic Inference
```powershell
python step5_test_inference.py
```
> **What Happens:** Confirms that the pipeline can load the saved LoRA weights from disk without crashing.

### Step 6: Sanity Check RAG
```powershell
python step6_rag_sanity_test.py
```
> **What Happens:** Tests the connection to your Google Gemini API key to ensure the Reasoning Agents are active and communicating.

### Step 7: The Full Agentic RAG Run
This is the core of the new architecture!
```powershell
python step7_rag_full_run.py
```
> **What Happens:** Data passes through the 8-Stage Flow:
> 1. **Fast Triage Agent** checks for easy cases.
> 2. **Clinical Context Agent** (Text only) extracts data.
> 3. **Image Analysis Agent** (Vision only) extracts data.
> 4. **Knowledge Retrieval Agent** searches the medical vector database.
> 5. **Asymmetric Synthesizer** (Gemini) merges everything to create a final, highly accurate diagnosis.

### Step 8: Final Evaluation
```powershell
python step8_evaluate.py
```
> **What Happens:** Scores your predictions against the official ground-truth answers and provides your final accuracy metrics. This is your definitive competition score!
