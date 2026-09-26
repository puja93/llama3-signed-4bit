#!/bin/bash
set -e

mkdir -p models
echo "================================================================================"
echo " Downloading LLaMA-3 Signed 4-Bit Model Bundle from Hugging Face..."
echo " Repo: https://huggingface.co/puja/llama3-zeromult-1b"
echo "================================================================================"

BASE_URL="https://huggingface.co/puja/llama3-zeromult-1b/resolve/main/models"

echo "[1/2] Downloading tokenizer.json (8.7 MB)..."
curl -L -o models/tokenizer.json "${BASE_URL}/tokenizer.json"

echo "[2/2] Downloading llama3_signed_4bit_b128.bin (626.3 MB)..."
curl -L -o models/llama3_signed_4bit_b128.bin "${BASE_URL}/llama3_signed_4bit_b128.bin"

echo ""
echo "================================================================================"
echo " Download finished successfully! All model files are ready in ./models/"
echo " To build and start chatting, run:"
echo "   make && ./build/chat"
echo "================================================================================"
