# Llama-3-Signed-4Bit

High-performance, ultra-compact C++ inference engine for Llama 3 (1B) using Signed 4-Bit Block-128 quantization on Apple Silicon.

## Model Origin

- **Base Model:** [Meta Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (16 Transformer layers, 2048 hidden dim, 32 attention heads, 128k vocabulary).
- **Quantization:** Converted from official BF16 safetensors into a custom Block-128 affine format. Every 128 weights are stored as 4-bit unsigned nibbles (64 bytes) with 2-byte scale and 2-byte min offset (4.25 bits/weight effective).
- **Hugging Face Hub:** Hosted at [`puja/llama3-zeromult-1b`](https://huggingface.co/puja/llama3-zeromult-1b).

## Key Features

- **626 MB Model Footprint**: Compact affine min-max quantization format with zero weight expansion in RAM.
- **25+ Tokens/Sec**: Factored GEMV arithmetic (`min * sum(x) + step * sum(q * x)`) with 4-row parallel ARM NEON vectorization.
- **Zero Dependencies**: Pure C++20 with Apple Accelerate/NEON and `mmap` zero-copy weight streaming.
- **High Fidelity**: Retains >0.993 cosine similarity to full precision.

## Quick Start

### 1. Download Model Bundle
Download the signed 4-bit model directly from Hugging Face:
```bash
./download_model.sh
```
Or manually fetch the files into `./models/`:
- **Model (626 MB)**: [llama3_signed_4bit_b128.bin](https://huggingface.co/puja/llama3-zeromult-1b/resolve/main/models/llama3_signed_4bit_b128.bin)
- **Tokenizer**: [tokenizer.json](https://huggingface.co/puja/llama3-zeromult-1b/resolve/main/models/tokenizer.json)

### 2. Build Engine
```bash
make
```

### 3. Run Inference
```bash
./build/chat
```
*(Optionally pass a custom model path: `./build/chat path/to/model.bin`)*

## Repository Structure
- `src/main.cpp`: Standalone C++20 inference engine with factored NEON GEMV.
- `download_model.sh`: Automated script to fetch weights and tokenizer from Hugging Face.
- `Makefile`: Build configuration for Apple Silicon (`-mcpu=apple-m1`).
- `models/`: Directory where downloaded model and tokenizer files are stored.
