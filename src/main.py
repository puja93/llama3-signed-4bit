#!/usr/bin/env python3
"""
LLaMA-3.2 1B Instruct - Apple Silicon Compact Signed 4-Bit Inference Engine
Pure In-Register ARM NEON SIMD GEMV via Zero-Overhead C Foreign Interface.
Modular Python Frontend delegating to src/llama_engine.
"""

import sys
import os
import time
import ctypes
import subprocess
from pathlib import Path

# ============================================================================
# Dynamic Library Loader & Auto-Compiler
# ============================================================================
# Resolve project root from src/main.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
BUILD_DIR = ROOT_DIR / "build"
DYLIB_PATH = BUILD_DIR / "libllama_engine.dylib"
SRC_CPP = SRC_DIR / "llama_engine.cpp"
SRC_H = SRC_DIR / "llama_engine.h"


def ensure_dylib_built() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not DYLIB_PATH.exists() or (
        SRC_CPP.exists() and SRC_CPP.stat().st_mtime > DYLIB_PATH.stat().st_mtime
    ):
        compile_cmd = [
            "clang++",
            "-O3",
            "-std=c++20",
            "-mcpu=apple-m1",
            "-dynamiclib",
            "-fPIC",
            str(SRC_CPP),
            "-o",
            str(DYLIB_PATH),
        ]
        res = subprocess.run(compile_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to build {DYLIB_PATH}:\n{res.stderr}")
    return DYLIB_PATH


# ============================================================================
# Ctypes Engine Wrapper
# ============================================================================
class LlamaEngineBindings:
    def __init__(self, dylib_path: Path):
        self.lib = ctypes.CDLL(str(dylib_path))

        # Engine bindings
        self.lib.llama_engine_create.argtypes = [ctypes.c_char_p]
        self.lib.llama_engine_create.restype = ctypes.c_void_p

        self.lib.llama_engine_free.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_free.restype = None

        self.lib.llama_engine_reset.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_reset.restype = None

        self.lib.llama_engine_get_num_layers.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_num_layers.restype = ctypes.c_int32

        self.lib.llama_engine_get_dim.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_dim.restype = ctypes.c_int32

        self.lib.llama_engine_get_ffn_dim.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_ffn_dim.restype = ctypes.c_int32

        self.lib.llama_engine_get_vocab_size.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_vocab_size.restype = ctypes.c_int32

        self.lib.llama_engine_get_num_heads.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_num_heads.restype = ctypes.c_int32

        self.lib.llama_engine_get_num_kv_heads.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_num_kv_heads.restype = ctypes.c_int32

        self.lib.llama_engine_get_head_dim.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_head_dim.restype = ctypes.c_int32

        self.lib.llama_engine_get_max_seq_len.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_max_seq_len.restype = ctypes.c_int32

        self.lib.llama_engine_forward_token.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.llama_engine_forward_token.restype = ctypes.c_int

        self.lib.llama_engine_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.llama_engine_prefill.restype = ctypes.c_int

        self.lib.llama_engine_sample.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.llama_engine_sample.restype = ctypes.c_int32

        self.lib.llama_engine_get_logits.argtypes = [ctypes.c_void_p]
        self.lib.llama_engine_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.llama_engine_dequantize_embedding.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.llama_engine_dequantize_embedding.restype = None

        # Built-in Tokenizer bindings
        self.lib.llama_tokenizer_create.argtypes = [ctypes.c_char_p]
        self.lib.llama_tokenizer_create.restype = ctypes.c_void_p

        self.lib.llama_tokenizer_free.argtypes = [ctypes.c_void_p]
        self.lib.llama_tokenizer_free.restype = None

        self.lib.llama_tokenizer_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        self.lib.llama_tokenizer_encode.restype = ctypes.c_int

        self.lib.llama_tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.lib.llama_tokenizer_decode.restype = ctypes.c_char_p


class TokenizerWrapper:
    """Tokenizer supporting Hugging Face 'tokenizers' with C++ engine fallback."""

    def __init__(self, tokenizer_path: str, bindings: LlamaEngineBindings):
        self._hf_tok = None
        self._c_tok = None
        self._bindings = bindings

        # Try Hugging Face tokenizers first for maximum speed
        try:
            from tokenizers import Tokenizer
            self._hf_tok = Tokenizer.from_file(tokenizer_path)
        except Exception:
            self._hf_tok = None

        if self._hf_tok is None:
            self._c_tok = bindings.lib.llama_tokenizer_create(tokenizer_path.encode("utf-8"))
            if not self._c_tok:
                raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}")

    def encode(self, text: str) -> list[int]:
        if self._hf_tok is not None:
            return self._hf_tok.encode(text, add_special_tokens=False).ids

        buf = (ctypes.c_int32 * 4096)()
        n = self._bindings.lib.llama_tokenizer_encode(
            self._c_tok, text.encode("utf-8"), buf, 4096
        )
        return [buf[i] for i in range(n)]

    def decode(self, token_id: int) -> str:
        if self._hf_tok is not None:
            return self._hf_tok.decode([token_id], skip_special_tokens=False)

        res = self._bindings.lib.llama_tokenizer_decode(self._c_tok, token_id)
        if res:
            return res.decode("utf-8", errors="replace")
        return ""

    def __del__(self):
        if self._c_tok:
            self._bindings.lib.llama_tokenizer_free(self._c_tok)
            self._c_tok = None


class LlamaEngine:
    """High-performance LLaMA 3 Signed 4-Bit Apple Silicon Engine."""

    def __init__(self, model_path: str, bindings: LlamaEngineBindings):
        self.bindings = bindings
        self.handle = bindings.lib.llama_engine_create(model_path.encode("utf-8"))
        if not self.handle:
            raise RuntimeError(f"Failed to load model from {model_path}")

        self.num_layers = bindings.lib.llama_engine_get_num_layers(self.handle)
        self.dim = bindings.lib.llama_engine_get_dim(self.handle)
        self.ffn_dim = bindings.lib.llama_engine_get_ffn_dim(self.handle)
        self.vocab_size = bindings.lib.llama_engine_get_vocab_size(self.handle)
        self.num_heads = bindings.lib.llama_engine_get_num_heads(self.handle)
        self.num_kv_heads = bindings.lib.llama_engine_get_num_kv_heads(self.handle)
        self.head_dim = bindings.lib.llama_engine_get_head_dim(self.handle)
        self.max_seq_len = bindings.lib.llama_engine_get_max_seq_len(self.handle)

    def reset(self):
        self.bindings.lib.llama_engine_reset(self.handle)

    def forward_token(self, token_id: int, pos: int, compute_logits: bool = True):
        self.bindings.lib.llama_engine_forward_token(
            self.handle, token_id, pos, 1 if compute_logits else 0
        )

    def prefill(self, tokens: list[int], start_pos: int, compute_last_logits: bool = False):
        if not tokens:
            return
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        self.bindings.lib.llama_engine_prefill(
            self.handle, arr, len(tokens), start_pos, 1 if compute_last_logits else 0
        )

    def sample(self, temperature: float = 0.0, top_p: float = 0.9) -> int:
        return self.bindings.lib.llama_engine_sample(self.handle, temperature, top_p)

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            self.bindings.lib.llama_engine_free(self.handle)
            self.handle = None


# ============================================================================
# Main Interactive Engine (Mimicking src/main.cpp)
# ============================================================================
def main():
    # Resolve default paths relative to workspace or cwd
    default_model = ROOT_DIR / "models" / "llama3_signed_4bit_b128.bin"
    default_tokenizer = ROOT_DIR / "models" / "tokenizer.json"

    model_path = str(default_model)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        model_path = sys.argv[1]

    print("\033[1;36m", end="")
    print("================================================================================")
    print("  LLaMA-3.2 1B INSTRUCT - APPLE SILICON COMPACT SIGNED 4-BIT ENGINE")
    print("  macOS (Apple Silicon M-Series) | ARM NEON High-Throughput SIMD (Python)")
    print("  Compact 4.25 Bits/Weight (626 MB Model) | Direct Fast Memory-Map")
    print("================================================================================\033[0m")

    # [1/2] Tokenizer
    tokenizer_path = str(default_tokenizer)
    print(f"[1/2] Loading Tokenizer from {tokenizer_path}...")
    if not os.path.exists(tokenizer_path):
        print(f"Error: Failed to find {tokenizer_path}", file=sys.stderr)
        return 1

    # Ensure native dylib is ready
    dylib = ensure_dylib_built()
    bindings = LlamaEngineBindings(dylib)

    try:
        tokenizer = TokenizerWrapper(tokenizer_path, bindings)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        return 1

    # [2/2] Model
    print(f"[2/2] Memory-Mapping Model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Cannot open {model_path}", file=sys.stderr)
        return 1

    t_start_load = time.perf_counter()
    try:
        engine = LlamaEngine(model_path, bindings)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    load_sec = time.perf_counter() - t_start_load

    def reset_history() -> list[int]:
        engine.reset()
        init_hist = [
            128000,  # <|begin_of_text|>
            128006,  # <|start_header_id|>
        ]
        init_hist.extend(tokenizer.encode("system"))
        init_hist.append(128007)  # <|end_header_id|>
        init_hist.extend(tokenizer.encode("\n\n"))
        init_hist.extend(tokenizer.encode("You are a helpful, concise, and smart AI assistant."))
        init_hist.append(128009)  # <|eot_id|>
        return init_hist

    print("\n================================================================================")
    print(f"  Model Loaded & Ready in {load_sec:.3f}s! (626.34 MB Model Active)")
    print("  Context Window: 8,192 Tokens Active | Pure In-Register SIMD GEMV.")
    print("  Type your prompt and press Enter.")
    print("  Commands: '/reset' or '/clear' to reset chat, '/exit' or '/quit' to leave.")
    print("================================================================================\n")

    chat_history = reset_history()
    processed_pos = 0

    while True:
        try:
            print("\033[1;32mYou>\033[0m ", end="", flush=True)
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            break

        if user_input in ("/reset", "/clear"):
            chat_history = reset_history()
            processed_pos = 0
            print("\033[1;33mChat history reset.\033[0m\n")
            continue

        # Format user prompt
        chat_history.append(128006)  # <|start_header_id|>
        chat_history.extend(tokenizer.encode("user"))
        chat_history.append(128007)  # <|end_header_id|>
        chat_history.extend(tokenizer.encode("\n\n"))
        chat_history.extend(tokenizer.encode(user_input))
        chat_history.append(128009)  # <|eot_id|>

        # Assistant header
        chat_history.append(128006)  # <|start_header_id|>
        chat_history.extend(tokenizer.encode("assistant"))
        chat_history.append(128007)  # <|end_header_id|>
        chat_history.extend(tokenizer.encode("\n\n"))

        # Fast prefill of prompt tokens up to the last one
        prefill_count = len(chat_history) - 1 - processed_pos
        if prefill_count > 0:
            engine.prefill(
                chat_history[processed_pos : len(chat_history) - 1],
                start_pos=processed_pos,
                compute_last_logits=False,
            )
            processed_pos = len(chat_history) - 1

        print("\033[1;35mLlama3-Signed-626MB>\033[0m ", end="", flush=True)

        t0_gen = time.perf_counter()
        gen_tokens = 0

        while processed_pos < engine.max_seq_len:
            engine.forward_token(chat_history[processed_pos], processed_pos, compute_logits=True)
            next_token = engine.sample(temperature=0.0, top_p=0.9)  # Greedy

            if next_token in (128001, 128009):  # <|end_of_text|> or <|eot_id|>
                processed_pos += 1
                break

            piece = tokenizer.decode(next_token)
            print(piece, end="", flush=True)

            chat_history.append(next_token)
            processed_pos += 1
            gen_tokens += 1

            if gen_tokens >= 512:
                break

        elapsed_sec = time.perf_counter() - t0_gen
        tok_s = (gen_tokens / elapsed_sec) if elapsed_sec > 0.0 else 0.0

        print(
            f"\n\033[0;37m[{gen_tokens} tokens | {tok_s:.1f} tok/s | "
            f"Compact Signed 4-Bit (626MB)]\033[0m\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
