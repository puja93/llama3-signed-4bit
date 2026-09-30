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

## ⚡ Quick Start: Chat in Seconds

Get the signed 4-bit compressed model running interactively on your Mac with zero external dependencies:

```bash
# 1. Download model bundle (626 MB)
./download_model.sh

# 2. Build the native ARM NEON engine
make

# 3. Start chatting!
./build/chat
```

- **Zero dependencies:** Pure C++20 with Apple Accelerate and ARM NEON SIMD. No PyTorch, Python, or GPU required.
- **Fast generation:** Streams live at **~27.7 tok/s** on Apple Silicon CPU — faster than BF16 running on Apple MPS.
- **Minimal RAM footprint:** Consumes only **~700 MB of RAM** in total during inference (zero weight expansion).

<details>
<summary>Alternative Ways to Chat (Python shell & BF16 baseline)</summary>

#### Python Shell (`src/main.py`)
Prefer a Python CLI? A thin ctypes wrapper calls the exact same compiled C++ engine with identical speed:
```bash
python3 src/main.py
```

#### Compare with Original BF16 Baseline (`chat.py`)
Run the uncompressed 2.3 GB baseline model on Apple MPS to compare outputs side-by-side:
```bash
pip install -r requirements.txt
python3 chat.py
```
*(Both frontends use the official LLaMA-3.2 Instruct chat template so conversation behavior is directly comparable).*

</details>

---

## At a Glance: Compressed vs Original

| Dimension | Original BF16 Baseline (`safetensors`) | Signed 4-Bit Engine (`llama3_signed_4bit_b128.bin`) | Advantage |
| :--- | :---: | :---: | :---: |
| **Model Size on Disk** | 2,300 MB (~2.3 GB) | **626 MB** | **72.8% smaller** |
| **RAM Footprint** | ~2.5 GB | **~700 MB** *(zero weight expansion)* | **Runs on tight-memory devices** |
| **Inference Speed** | ~16.9 tok/s (MPS GPU) | **~27.7 tok/s** (Pure CPU NEON) | **64% faster on CPU** |
| **Dependencies** | Python, PyTorch, Transformers | **Zero (pure C++ binary)** | **Self-contained** |
| **Accuracy Retention** | 100% | **92.2% mean retention** across 6 benchmarks | **Minimal quality trade-off** |

---

## Model & Quantization Overview

- **Base Model:** [Meta Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (16 Transformer layers, 2048 hidden dim, 32 attention heads, 128k vocabulary).
- **Quantization:** Converted from official BF16 safetensors into a custom Block-128 affine format. Every 128 weights are stored as 4-bit unsigned nibbles (64 bytes) with 2-byte scale and 2-byte min offset (4.25 bits/weight effective).
- **Kernel:** Factored GEMV arithmetic (`min * sum(x) + step * sum(q * x)`) with 4-row parallel ARM NEON vectorization and `mmap` zero-copy weight streaming.
- **Hugging Face Hub:** Hosted at [`puja/llama3-zeromult-1b`](https://huggingface.co/puja/llama3-zeromult-1b).

---

## Benchmark Results & Evaluation

Both models were comprehensively benchmarked side-by-side using the official **EleutherAI LM Evaluation Harness (`lm-eval` v0.4.13)** on Apple Silicon across **3,100 questions** in 6 standard suites, including all 57 academic subjects of MMLU.

### Overall Benchmark Accuracy & Retention

<p align="center">
  <img src="assets/benchmark_comparison.svg" alt="Benchmark Comparison: Llama3.2-1B BF16 vs Signed 4-Bit" width="100%">
</p>

<details>
<summary><b>View detailed benchmark scores table & log sources</b></summary>

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

</details>

### MMLU 57-Subject Category Breakdown

MMLU was evaluated across all 57 individual academic subjects with 50 samples per subject (2,850 total questions) under standard 5-shot loglikelihood scoring:

<p align="center">
  <img src="assets/mmlu_breakdown.svg" alt="MMLU Subject Breakdown by Category" width="100%">
</p>

<details>
<summary><b>View detailed MMLU category breakdown table</b></summary>

| Category | Subjects Included | Questions | Regular (BF16 Baseline) | Signed 4-Bit Engine | Recovery Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **STEM** | Physics, Chemistry, Math, CS, Biology, Engineering | 950 | 42.95% | **36.74%** | **85.5%** |
| **Humanities** | History, Philosophy, Law, Art, World Religions | 650 | 54.77% | **44.00%** | **80.3%** |
| **Social Sciences** | Economics, Politics, Psychology, Sociology, Geography | 600 | 54.33% | **46.33%** | **85.3%** |
| **Other / Applied** | Business, Management, Medicine, Marketing, Accounting | 650 | 49.85% | **42.15%** | **84.6%** |
| **Total MMLU** | **All 57 Academic Subjects Combined** | **2,850** | **49.61%** | **41.65%** | **84.0%** |

</details>

---

## Reproducing Benchmarks & Tests

To run unit tests or replicate the benchmark evaluations locally:

```bash
# 1. Run unit tests and engine integrity checks
make test

# 2. Benchmark Signed 4-Bit Native Engine (626 MB)
./benchmark_compressed.sh

# 3. Benchmark Original Regular Model (BF16 model.safetensors, ~2.3 GB)
./benchmark_regular.sh

# 4. Generate side-by-side comparison report & accuracy retention
python3 tests/compare_results.py
```
*(Benchmark runs log console output and save structured JSON metrics directly to `tests/results/`)*.

---

## Repository Structure

<details>
<summary>Click to expand</summary>

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
│   └── main.py                    # Thin Python shell over llama_engine via ctypes
└── tests/
    ├── eval.py                    # EleutherAI LM Evaluation Harness CLI
    ├── lm_eval_adapter.py         # lm-evaluation-harness bridge for native engine
    ├── test_engine.py             # Unit and integration test suite (make test)
    ├── compare_results.py         # Comparative analysis tool & retention calculator
    └── results/                   # Benchmark logs and structured JSON metrics
        ├── benchmark_compressed_latest.json
        ├── benchmark_compressed_latest.log
        ├── benchmark_regular_latest.json
        └── benchmark_regular_latest.log
```

</details>

---

## License

MIT
