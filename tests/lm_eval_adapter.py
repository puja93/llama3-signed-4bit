#!/usr/bin/env python3
"""
EleutherAI lm-evaluation-harness Model Adapter for LLaMA 3 Signed 4-Bit Apple Silicon Engine.
Enables running official Open LLM Leaderboard benchmarks.
"""

import sys
from typing import List, Tuple, Any, Optional, Iterator
import numpy as np
from pathlib import Path

# Add project root to sys.path
TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

from src.main import (
    ensure_dylib_built,
    LlamaEngineBindings,
    LlamaEngine,
    TokenizerWrapper,
)


@register_model("signed_4bit", "llama3_signed_4bit")
class Signed4BitLM(LM):
    """
    Adapter subclassing lm_eval.api.model.LM to evaluate Signed 4-Bit LLaMA on
    standard benchmarks (ARC, HellaSwag, MMLU, GSM8K, WikiText, etc.).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        batch_size: int = 1,
        max_length: int = 8192,
        max_gen_toks: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)
        self._max_gen_toks = int(max_gen_toks)
        self._device = device

        dylib = ensure_dylib_built()
        self.bindings = LlamaEngineBindings(dylib)

        if not model_path:
            model_path = str(ROOT_DIR / "models" / "llama3_signed_4bit_b128.bin")
        if not tokenizer_path:
            tokenizer_path = str(ROOT_DIR / "models" / "tokenizer.json")

        self.engine = LlamaEngine(model_path, self.bindings)
        self.tokenizer = TokenizerWrapper(tokenizer_path, self.bindings)

    @property
    def eot_token_id(self) -> int:
        return 128009  # <|eot_id|>

    @property
    def prefix_token_id(self) -> int:
        return 128000  # <|begin_of_text|>

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    def tok_encode(
        self,
        string: str,
        add_special_tokens: Optional[bool] = False,
        left_truncate_len: Optional[int] = None,
        **kwargs,
    ) -> List[int]:
        tokens = self.tokenizer.encode(string)
        if add_special_tokens:
            tokens = [self.prefix_token_id] + tokens
        if left_truncate_len is not None and len(tokens) > left_truncate_len:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(
        self,
        tokens: Iterator[List[int]],
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(tokens, int):
            return self.tokenizer.decode(tokens)
        return "".join(self.tokenizer.decode(t) for t in tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Computes log-likelihood of continuation given context: log P(continuation | context).
        Optimized with KV prefix caching: consecutive requests sharing identical context
        reuse the prefilled KV cache and initial logits, avoiding 4x redundant prompt prefill!
        """
        results = []
        cached_ctx_tokens = None
        cached_init_logits = None

        for req in requests:
            context, continuation = req.args

            if not context:
                ctx_tokens = [self.prefix_token_id]
            else:
                ctx_tokens = self.tok_encode(context, add_special_tokens=False)
                if not ctx_tokens:
                    ctx_tokens = [self.prefix_token_id]

            cont_tokens = self.tok_encode(continuation, add_special_tokens=False)
            if not cont_tokens:
                results.append((0.0, True))
                continue

            all_len = len(ctx_tokens) + len(cont_tokens)
            if all_len > self.max_length:
                keep_ctx = max(1, self.max_length - len(cont_tokens))
                ctx_tokens = ctx_tokens[-keep_ctx:]

            # Check if context matches cached prefix in the KV cache
            is_cache_hit = (cached_ctx_tokens is not None and ctx_tokens == cached_ctx_tokens)

            if not is_cache_hit:
                # Reset position to 0 and prefill the new context
                self.engine.reset()
                if len(ctx_tokens) > 1:
                    self.engine.prefill(
                        ctx_tokens[:-1], start_pos=0, compute_last_logits=False
                    )

                # Forward the last context token to produce logits for continuation[0]
                pos = len(ctx_tokens) - 1
                self.engine.forward_token(ctx_tokens[-1], pos, compute_logits=True)

                # Cache context and its resulting logits
                cached_ctx_tokens = list(ctx_tokens)
                ptr = self.bindings.lib.llama_engine_get_logits(self.engine.handle)
                cached_init_logits = np.ctypeslib.as_array(ptr, shape=(self.engine.vocab_size,)).copy()

            sum_logprob = 0.0
            is_greedy = True
            pos = len(ctx_tokens) - 1

            for i, target_tok in enumerate(cont_tokens):
                if i == 0:
                    logits = cached_init_logits
                else:
                    ptr = self.bindings.lib.llama_engine_get_logits(self.engine.handle)
                    logits = np.ctypeslib.as_array(ptr, shape=(self.engine.vocab_size,))

                max_l = np.max(logits)
                log_probs = (logits - max_l) - np.log(np.sum(np.exp(logits - max_l)))

                logprob = float(log_probs[target_tok])
                sum_logprob += logprob

                if int(np.argmax(logits)) != int(target_tok):
                    is_greedy = False

                # Forward this token to get logits for the next continuation token
                if i + 1 < len(cont_tokens):
                    pos += 1
                    self.engine.forward_token(target_tok, pos, compute_logits=True)

            results.append((sum_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """
        Computes rolling log-likelihood of text string for perplexity evaluation (e.g. WikiText).
        """
        results = []
        for req in requests:
            (string,) = req.args
            tokens = self.tok_encode(string, add_special_tokens=False)
            if not tokens:
                results.append(0.0)
                continue

            tokens = [self.prefix_token_id] + tokens
            sum_logprob = 0.0
            chunk_size = self.max_length

            for start_idx in range(0, len(tokens) - 1, chunk_size - 1):
                end_idx = min(start_idx + chunk_size, len(tokens))
                chunk = tokens[start_idx:end_idx]
                if len(chunk) <= 1:
                    break

                self.engine.reset()
                self.engine.forward_token(chunk[0], 0, compute_logits=True)

                for pos in range(1, len(chunk)):
                    target_tok = chunk[pos]
                    ptr = self.bindings.lib.llama_engine_get_logits(self.engine.handle)
                    logits = np.ctypeslib.as_array(ptr, shape=(self.engine.vocab_size,))

                    max_l = np.max(logits)
                    log_probs = (logits - max_l) - np.log(np.sum(np.exp(logits - max_l)))
                    sum_logprob += float(log_probs[target_tok])

                    if pos + 1 < len(chunk):
                        self.engine.forward_token(target_tok, pos, compute_logits=True)

            results.append(float(sum_logprob))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generates text until stop sequences are reached (e.g. GSM8K).
        """
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_tokens = self.tok_encode(context, add_special_tokens=False)
            if not ctx_tokens:
                ctx_tokens = [self.prefix_token_id]

            self.engine.reset()
            if len(ctx_tokens) > 1:
                self.engine.prefill(
                    ctx_tokens[:-1], start_pos=0, compute_last_logits=False
                )

            pos = len(ctx_tokens) - 1
            self.engine.forward_token(ctx_tokens[-1], pos, compute_logits=True)

            generated_text = ""
            for _ in range(max_gen_toks):
                next_token = self.engine.sample(temperature=0.0, top_p=0.9)
                if next_token in (128001, 128009):
                    break

                piece = self.tokenizer.decode(next_token)
                generated_text += piece

                # Check stop sequences
                stop_hit = False
                for s in until:
                    if s and s in generated_text:
                        generated_text = generated_text[: generated_text.index(s)]
                        stop_hit = True
                        break
                if stop_hit:
                    break

                pos += 1
                if pos >= self.max_length:
                    break
                self.engine.forward_token(next_token, pos, compute_logits=True)

            results.append(generated_text)

        return results
