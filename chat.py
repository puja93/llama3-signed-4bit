#!/opt/anaconda3/bin/python3
"""
LLaMA-3.2 Instruct Interactive Terminal Chat Application
Meta Official Pipeline (Hugging Face / Meta Llama Cookbook Standard)
"""

import os
import sys

# Auto-re-exec with Anaconda environment if torch is not in current Python
try:
    import torch
    import transformers
except ImportError:
    anaconda_py = "/opt/anaconda3/bin/python3"
    if os.path.exists(anaconda_py) and sys.executable != anaconda_py:
        os.execv(anaconda_py, [anaconda_py] + sys.argv)
    print("\n\033[1;31m[Environment Notice] Missing PyTorch in current Python:\033[0m")
    print("Please run with:")
    print("    \033[1;32m/opt/anaconda3/bin/python3 chat.py\033[0m\n")
    sys.exit(1)

import mmap
import struct
import time
import numpy as np
import warnings
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# Suppress unnecessary verbose logging
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_hf_model_from_signed_bin(bin_path: str, model_dir: str, device: str, dtype: torch.dtype):
    """Populates Hugging Face AutoModelForCausalLM directly from the 626 MB raw binary (NO ZIP)."""
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    model.model.rotary_emb = LlamaRotaryEmbedding(config=config, device=device)

    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    offset = 36  # magic (8) + 7 uint32 config (28) = 36
    final_norm = np.frombuffer(mm, dtype=np.float32, count=2048, offset=offset)
    offset += 2048 * 4

    def set_param(module, name, tensor):
        delattr(module, name)
        setattr(module, name, torch.nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False))

    set_param(model.model.norm, "weight", torch.from_numpy(final_norm.copy()))

    def unpack_matrix(buf, off):
        out_dim, in_dim, num_blocks = struct.unpack_from("<3I", buf, off)
        off += 12
        raw_nib = np.frombuffer(buf, dtype=np.uint8, count=num_blocks * 64, offset=off).reshape(num_blocks, 64)
        off += num_blocks * 64
        step_u16 = np.frombuffer(buf, dtype=np.uint16, count=num_blocks, offset=off)
        off += num_blocks * 2
        min_u16 = np.frombuffer(buf, dtype=np.uint16, count=num_blocks, offset=off)
        off += num_blocks * 2

        steps = (step_u16.astype(np.uint32) << 16).view(np.float32)[:, np.newaxis]
        mins = (min_u16.astype(np.uint32) << 16).view(np.float32)[:, np.newaxis]
        q0 = raw_nib & 0x0F
        q1 = (raw_nib >> 4) & 0x0F
        q_all = np.empty((num_blocks, 128), dtype=np.float32)
        q_all[:, 0::2] = q0
        q_all[:, 1::2] = q1
        w = q_all * steps + mins
        return torch.from_numpy(w.reshape(out_dim, in_dim)), off

    for l in range(16):
        in_norm = np.frombuffer(mm, dtype=np.float32, count=2048, offset=offset)
        offset += 2048 * 4
        post_norm = np.frombuffer(mm, dtype=np.float32, count=2048, offset=offset)
        offset += 2048 * 4
        layer = model.model.layers[l]
        set_param(layer.input_layernorm, "weight", torch.from_numpy(in_norm.copy()))
        set_param(layer.post_attention_layernorm, "weight", torch.from_numpy(post_norm.copy()))

        w_q, offset = unpack_matrix(mm, offset)
        set_param(layer.self_attn.q_proj, "weight", w_q)
        w_k, offset = unpack_matrix(mm, offset)
        set_param(layer.self_attn.k_proj, "weight", w_k)
        w_v, offset = unpack_matrix(mm, offset)
        set_param(layer.self_attn.v_proj, "weight", w_v)
        w_o, offset = unpack_matrix(mm, offset)
        set_param(layer.self_attn.o_proj, "weight", w_o)

        w_gate, offset = unpack_matrix(mm, offset)
        set_param(layer.mlp.gate_proj, "weight", w_gate)
        w_up, offset = unpack_matrix(mm, offset)
        set_param(layer.mlp.up_proj, "weight", w_up)
        w_down, offset = unpack_matrix(mm, offset)
        set_param(layer.mlp.down_proj, "weight", w_down)

    w_embed, offset = unpack_matrix(mm, offset)
    set_param(model.model.embed_tokens, "weight", w_embed)
    model.lm_head.weight = model.model.embed_tokens.weight

    model.eval()
    return model


import argparse

def main():
    parser = argparse.ArgumentParser(description="LLaMA-3.2 1B Instruct Chat & Benchmark")
    parser.add_argument("--original", action="store_true", default=False, help="Explicitly load original unquantized BF16 model (model.safetensors)")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", default=False, help="Load signed 4-bit raw binary (llama3_signed_4bit_b128.bin)")
    parser.add_argument("--prompt", type=str, default=None, help="Run single prompt and exit (benchmark mode)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default=None, help="Target device: mps, cpu, cuda")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    if not os.path.exists(model_dir):
        model_dir = os.path.abspath("models")

    # Determine whether to load original or 4-bit:
    # If --original is set, or if --4bit is NOT set and model.safetensors exists, prefer original for chat.py
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    signed_bin = os.path.join(model_dir, "llama3_signed_4bit_b128.bin")
    
    use_4bit = False
    if args.four_bit:
        use_4bit = True
    elif args.original:
        use_4bit = False
    elif os.path.exists(safetensors_path):
        use_4bit = False  # Default to original 1B BF16 when model.safetensors is available
    elif os.path.exists(signed_bin):
        use_4bit = True

    model_title = "Signed 4-Bit Raw Binary (626 MB)" if use_4bit else "Original LLaMA-3.2-1B (BF16 model.safetensors, ~2.3 GB)"

    print("\033[1;36m" + "=" * 80)
    print("  LLaMA-3.2 1B INSTRUCT - INTERACTIVE TERMINAL CHAT")
    print(f"  Mode: {model_title} | Hugging Face Pipeline")
    print("=" * 80 + "\033[0m\n")

    # 1. Device Selection (Apple Silicon MPS / CUDA / CPU)
    if args.device:
        device = args.device
        dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
        print(f"\033[1;33m[1/3] Device (User): {device} | Dtype: {dtype}\033[0m")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
        print("\033[1;33m[1/3] Device: Apple Silicon Metal (MPS) | Dtype: bfloat16\033[0m")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("\033[1;33m[1/3] Device: NVIDIA CUDA | Dtype: bfloat16\033[0m")
    else:
        device = "cpu"
        dtype = torch.float32
        print("\033[1;33m[1/3] Device: CPU | Dtype: float32\033[0m")

    # 2. Load Tokenizer
    print(f"\033[1;33m[2/3] Loading Tokenizer from {model_dir}...\033[0m")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception as e:
        print(f"\033[1;31mError loading tokenizer: {e}\033[0m")
        sys.exit(1)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    ]

    # 3. Load Model into Memory
    if use_4bit:
        mb_size = os.path.getsize(signed_bin) / (1024 * 1024)
        print(f"\033[1;33m[3/3] Loading Model from Signed 4-Bit Raw Binary ({mb_size:.1f} MB, NO ZIP)...\033[0m")
        t0 = time.perf_counter()
        try:
            model = load_hf_model_from_signed_bin(signed_bin, model_dir, device, dtype)
            print(f"\033[1;32m • Hugging Face model loaded in {time.perf_counter() - t0:.2f}s!\033[0m")
        except Exception as e:
            print(f"\033[1;31mError loading signed binary: {e}\033[0m")
            sys.exit(1)
    else:
        print(f"\033[1;33m[3/3] Loading Original Model from {safetensors_path}...\033[0m")
        t0 = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True
            )
            model.eval()
            print(f"\033[1;32m • Original HF model loaded in {time.perf_counter() - t0:.2f}s!\033[0m")
        except Exception as e:
            print(f"\033[1;31mError loading model: {e}\033[0m")
            sys.exit(1)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    system_prompt = "You are a helpful, smart, and friendly AI assistant."

    # If single prompt benchmark mode:
    if args.prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.prompt}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(device)

        print("\033[1;32mLlama3-Instruct>\033[0m ", end="", flush=True)
        t_gen_0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=args.max_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id
            )
        t_gen = time.perf_counter() - t_gen_0
        num_gen = outputs[0].shape[0] - inputs["input_ids"].shape[1]
        tok_s = num_gen / t_gen if t_gen > 0 else 0
        print(f"\n\033[1;34m[Stats: {num_gen} tokens in {t_gen:.2f}s = {tok_s:.2f} tok/s]\033[0m\n")
        return

    print("\n\033[1;32m" + "=" * 80)
    print("  Model Loaded & Ready in RAM! Live chat session active.")
    print("  Type your prompt and press Enter.")
    print("  Commands: '/reset' or '/clear' to reset chat, '/exit' or '/quit' to leave.")
    print("=" * 80 + "\033[0m\n")

    conversation_history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\033[1;34mYou>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        if user_input.lower() in ["/reset", "/clear", "clear"]:
            conversation_history = [{"role": "system", "content": system_prompt}]
            print("\033[1;33m[Conversation history reset]\033[0m\n")
            continue

        # Append User Message
        conversation_history.append({"role": "user", "content": user_input})

        # Apply Meta Official Chat Template (apply_chat_template)
        try:
            inputs = tokenizer.apply_chat_template(
                conversation_history,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)
        except Exception as e:
            print(f"\033[1;31mTemplate error: {e}\033[0m")
            continue

        print("\033[1;32mLlama3-Instruct>\033[0m ", end="", flush=True)

        try:
            t_gen_0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=args.max_tokens,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id
                )
            t_gen = time.perf_counter() - t_gen_0

            # Extract generated response text to keep multi-turn context
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            num_gen = len(generated_tokens)
            tok_s = (num_gen / t_gen) if t_gen > 0 else 0.0
            print(f"\n\033[1;34m[Stats: {num_gen} tokens in {t_gen:.2f}s = {tok_s:.2f} tok/s]\033[0m\n")

            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            conversation_history.append({"role": "assistant", "content": response_text})

        except Exception as e:
            print(f"\n\033[1;31mGeneration error: {e}\033[0m\n")

if __name__ == "__main__":
    main()
