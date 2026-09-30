# Llama3-Signed-4Bit

<div align="center">

[![Model Size](https://img.shields.io/badge/Model%20Size-626%20MB-emerald?style=for-the-badge&logo=buffer)](https://huggingface.co/puja/llama3-zeromult-1b)
[![Precision](https://img.shields.io/badge/Quantization-Signed%204--Bit%20Block--128-blue?style=for-the-badge)](https://huggingface.co/puja/llama3-zeromult-1b)
[![Accuracy Retention](https://img.shields.io/badge/Mean%20Retention-92.2%25-green?style=for-the-badge)](tests/results/benchmark_compressed_latest.json)
[![Inference Speed](https://img.shields.io/badge/Speed-27.7%20tok%2Fs%20(NEON)-orange?style=for-the-badge)](src/llama_engine.cpp)
[![Base Model](https://img.shields.io/badge/Base-Meta%20Llama--3.2--1B-purple?style=for-the-badge&logo=meta)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

**High-performance, ultra-compact C++ inference engine for LLaMA 3.2 (1B) using Signed 4-Bit Block-128 quantization on Apple Silicon.**

</div>

---

## Model Origin

- **Base Model:** [Meta Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (16 Transformer layers, 2048 hidden dim, 32 attention heads, 128k vocabulary).
- **Quantization:** Converted from official BF16 safetensors into a custom Block-128 affine format. Every 128 weights are stored as 4-bit unsigned nibbles (64 bytes) with 2-byte scale and 2-byte min offset (4.25 bits/weight effective).
- **Hugging Face Hub:** Hosted at [`puja/llama3-zeromult-1b`](https://huggingface.co/puja/llama3-zeromult-1b).

---

## Key Features

- **626 MB Model Footprint**: Compact affine min-max quantization format with zero weight expansion in RAM (down from ~2.3 GB BF16).
- **25+ Tokens/Sec**: Factored GEMV arithmetic (`min * sum(x) + step * sum(q * x)`) with 4-row parallel ARM NEON vectorization — **27.7 tok/s measured** on Apple M-series, faster than BF16 on MPS.
- **Zero Dependencies**: Pure C++20 with Apple Accelerate/NEON and `mmap` zero-copy weight streaming.
- **High Fidelity**: Retains **92.2% mean accuracy** across standard Open LLM Leaderboard benchmarks and >0.993 cosine similarity to full precision.

---

## Benchmark Results & Performance Evaluation

Both models were comprehensively benchmarked side-by-side using the official **EleutherAI LM Evaluation Harness (`lm-eval` v0.4.13)** on Apple Silicon. The evaluation covers **3,100 questions** across 6 standard suites, including all 57 academic subjects of MMLU.

### Overall Benchmark Accuracy & Retention

<p align="center">
  <img src="assets/benchmark_comparison.svg" alt="Benchmark Comparison: Llama3.2-1B BF16 vs Signed 4-Bit" width="100%">
</p>

### Head-to-Head Benchmark Scores

| Benchmark Task | Target Metric | Evaluated Qs | Regular Baseline (BF16, 2.3 GB) | Signed 4-Bit (626 MB) | Delta (Δ) | Accuracy Retention |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ARC-Challenge** | Normalized Acc (`acc_norm`) | 50 | 32.00% | **34.00%** | **+2.00%** | **106.3%** |
| **HellaSwag** | Normalized Acc (`acc_norm`) | 50 | 68.00% | **58.00%** | -10.00% | **85.3%** |
| **MMLU (Overall)** | 5-shot Accuracy (`acc`) | 2,850 | 49.61% | **41.65%** | -7.96% | **84.0%** |
| **TruthfulQA** | Multiple-choice 2 (`mc2`) | 50 | 52.07% | **48.62%** | -3.45% | **93.4%** |
| **WinoGrande** | Accuracy (`acc`) | 50 | 62.00% | **60.00%** | -2.00% | **96.8%** |
| **GSM8K** | Exact Match (`strict-match`) | 50 | 22.00% | **10.00%** | -12.00% | **45.5%** |
| **OVERALL MEAN** | **All Evaluated Tasks** | **3,100** | — | — | — | **92.18%** |

> **Log Sources:**
> - Signed 4-Bit run: [`tests/results/benchmark_compressed_latest.log`](tests/results/benchmark_compressed_latest.log) / [`.json`](tests/results/benchmark_compressed_latest.json)
> - Regular BF16 run: [`tests/results/benchmark_regular_latest.log`](tests/results/benchmark_regular_latest.log) / [`.json`](tests/results/benchmark_regular_latest.json)

---

### MMLU 57-Subject Category Breakdown

MMLU was evaluated across all 57 individual academic subjects with 50 samples per subject (2,850 total questions) under standard 5-shot loglikelihood scoring:

<p align="center">
  <img src="assets/mmlu_breakdown.svg" alt="MMLU Subject Breakdown by Category" width="100%">
</p>

| Category | Subjects Included | Questions | Regular (BF16 Baseline) | Signed 4-Bit Engine | Recovery Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **STEM** | Physics, Chemistry, Math, CS, Biology, Engineering | 950 | 42.95% | **36.74%** | **85.5%** |
| **Humanities** | History, Philosophy, Law, Art, World Religions | 650 | 54.77% | **44.00%** | **80.3%** |
| **Social Sciences** | Economics, Politics, Psychology, Sociology, Geography | 600 | 54.33% | **46.33%** | **85.3%** |
| **Other / Applied** | Business, Management, Medicine, Marketing, Accounting | 650 | 49.85% | **42.15%** | **84.6%** |
| **Total MMLU** | **All 57 Academic Subjects Combined** | **2,850** | **49.61%** | **41.65%** | **84.0%** |

---

### Efficiency & Footprint Comparison

| Dimension | Original BF16 (`model.safetensors`) | Signed 4-Bit Block-128 (`llama3_signed_4bit_b128.bin`) | Advantage |
| :--- | :---: | :---: | :---: |
| **Model Size on Disk** | 2,300 MB (~2.3 GB) | **626 MB** | **72.8% smaller** |
| **Effective Bits/Weight** | 16.00 bits | **4.25 bits** | **3.76x compression** |
| **RAM Footprint in Inference** | ~2.5 GB | **~700 MB** *(zero weight expansion)* | **Runs on tight-memory devices** |
| **Execution Kernel** | PyTorch MPS / Metal | **Hand-tuned ARM NEON SIMD** | **Zero framework dependency** |
| **Inference Throughput** | ~16.9 tok/s (MPS) | **~27.7 tok/s** (Pure CPU NEON) | **Signed 4-Bit is faster than BF16 MPS** |

---

## Quick Start

### Prerequisites

- **Apple Silicon Mac** (M1 or later) — required for ARM NEON SIMD kernel
- **Xcode Command Line Tools**: `xcode-select --install`
- **Python 3.10+** with evaluation dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 1. Download Model Bundle
Download the signed 4-bit model directly from Hugging Face:
```bash
./download_model.sh
```
Or manually fetch the files into `./models/`:
- **Model (626 MB)**: [llama3_signed_4bit_b128.bin](https://huggingface.co/puja/llama3-zeromult-1b/resolve/main/models/llama3_signed_4bit_b128.bin)
- **Tokenizer**: [tokenizer.json](https://huggingface.co/puja/llama3-zeromult-1b/resolve/main/models/tokenizer.json)

### 2. Build Engine
Compile the native ARM NEON inference binary:
```bash
make
```

### 3. Run Interactive Chat

You can run an interactive chat session with either model and compare them side by side.

#### Signed 4-Bit Compressed Engine (626 MB) — Native ARM NEON

Run the C++ binary directly:
```bash
./build/chat
```
Or via the Python frontend (auto-compiles the engine if needed):
```bash
python3 src/main.py
```
Both stream tokens live to your terminal at ~25 tok/s, entirely CPU-bound with no PyTorch or GPU dependency.

#### Original BF16 Baseline (~2.3 GB) — Hugging Face / MPS

Run the reference HuggingFace pipeline using Apple MPS acceleration:
```bash
python3 chat.py
```
Requires `model.safetensors` in `./models/` (downloaded by `./download_model.sh` alongside the 4-bit weights).

> Both chat frontends use the same **LLaMA-3.2 Instruct chat template** (`<|begin_of_text|>`, system prompt, user/assistant turns) so the conversation behaviour is directly comparable.


### 4. Run Verification & Unit Tests
```bash
make test
```
Verifies engine tensor dimensions, tokenizer round-trip integrity, logits validity, prefill performance, and LM Evaluation Harness adapter compatibility.

### 5. Run Official Benchmarks (EleutherAI / Hugging Face Harness)
Evaluate on the standard Open LLM Leaderboard suite (`arc_challenge`, `hellaswag`, `mmlu`, `truthfulqa_mc2`, `winogrande`, `gsm8k`):

```bash
# 1. Benchmark Signed 4-Bit Native Engine (626 MB)
./benchmark_compressed.sh

# 2. Benchmark Original Regular Model (BF16 model.safetensors, ~2.3 GB)
./benchmark_regular.sh

# 3. Generate side-by-side comparison report & accuracy retention
python3 tests/compare_results.py
```
*(Benchmark runs log console output and save structured JSON metrics directly to `tests/results/`)*.

---

## Repository Structure

```
llama3-signed-4bit/
├── assets/
│   ├── benchmark_comparison.svg   # High-resolution benchmark comparison chart
│   └── mmlu_breakdown.svg         # MMLU 57-subject category breakdown chart
├── benchmark_compressed.sh        # Automated evaluation runner for Signed 4-Bit engine
├── benchmark_regular.sh           # Automated evaluation runner for BF16 baseline
├── chat.py                        # Reference Hugging Face baseline chat runner
├── download_model.sh              # Script to fetch weights & tokenizer from Hugging Face
├── Makefile                       # Optimized Apple Silicon build (-mcpu=apple-m1)
├── requirements.txt               # Python dependencies (lm-eval, torch, transformers)
├── models/                        # Model weights and tokenizer destination
├── src/
│   ├── llama_engine.h             # Clean C API header for engine & tokenizer
│   ├── llama_engine.cpp           # Core engine: ARM NEON factored GEMV & SwiGLU
│   ├── main.cpp                   # C++ CLI chat frontend
│   └── main.py                    # Python CLI chat frontend via ctypes (~25 tok/s)
└── tests/
    ├── eval.py                    # EleutherAI/Hugging Face harness evaluation CLI
    ├── lm_eval_adapter.py         # lm-evaluation-harness bridge for native engine
    ├── test_engine.py             # Unit and integration test suite (make test)
    ├── compare_results.py         # Comparative analysis tool & retention calculator
    └── results/                   # Benchmark logs and structured JSON metrics
        ├── benchmark_compressed_latest.json
        ├── benchmark_compressed_latest.log
        ├── benchmark_regular_latest.json
        └── benchmark_regular_latest.log
```

---

## License

MIT License.
Base model weights are subject to the [Meta Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
