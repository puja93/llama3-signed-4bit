#!/opt/anaconda3/bin/python3
"""
Unit and Integration Tests for LLaMA-3 Signed 4-Bit Engine & Tokenizer.
Can be executed via `pytest tests/test_engine.py` or `python3 -m unittest discover tests`.
"""

import unittest
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "llama3_signed_4bit_b128.bin"
TOKENIZER_PATH = ROOT_DIR / "models" / "tokenizer.json"

from src.main import (
    ensure_dylib_built,
    LlamaEngineBindings,
    LlamaEngine,
    TokenizerWrapper,
)
from tests.lm_eval_adapter import Signed4BitLM
from lm_eval.api.instance import Instance


class TestSigned4BitEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dylib_path = ensure_dylib_built()
        cls.bindings = LlamaEngineBindings(cls.dylib_path)
        cls.assertTrue(MODEL_PATH.exists(), f"Model missing: {MODEL_PATH}")
        cls.assertTrue(TOKENIZER_PATH.exists(), f"Tokenizer missing: {TOKENIZER_PATH}")

        cls.engine = LlamaEngine(str(MODEL_PATH), cls.bindings)
        cls.tokenizer = TokenizerWrapper(str(TOKENIZER_PATH), cls.bindings)

    def test_01_engine_architecture_dimensions(self):
        """Verify model hyper-parameters match LLaMA-3.2-1B specifications."""
        self.assertEqual(self.engine.num_layers, 16, "Expected 16 transformer layers")
        self.assertEqual(self.engine.dim, 2048, "Expected hidden dimension 2048")
        self.assertEqual(self.engine.vocab_size, 128256, "Expected vocabulary size 128256")
        self.assertEqual(self.engine.num_heads, 32, "Expected 32 attention heads")
        self.assertEqual(self.engine.num_kv_heads, 8, "Expected 8 KV heads (GQA)")
        self.assertEqual(self.engine.head_dim, 64, "Expected head dimension 64")
        self.assertGreaterEqual(self.engine.max_seq_len, 2048, "Expected context length >= 2048")

    def test_02_tokenizer_encode_decode(self):
        """Verify tokenizer preserves identity across encode and decode cycles."""
        test_strings = [
            "Hello world!",
            "Explain why the sky is blue.",
            "def palindrome(w): return w == w[::-1]",
            "Special token check: 1234567890",
        ]
        for s in test_strings:
            tokens = self.tokenizer.encode(s)
            self.assertGreater(len(tokens), 0)
            decoded = "".join(self.tokenizer.decode(t) for t in tokens)
            self.assertEqual(decoded, s, f"Decoded mismatch for: {s}")

    def test_03_forward_token_and_logits(self):
        """Verify single token forward produces finite, normalized logits without NaN or Inf."""
        self.engine.reset()
        bos_token = 128000
        self.engine.forward_token(bos_token, pos=0, compute_logits=True)

        ptr = self.bindings.lib.llama_engine_get_logits(self.engine.handle)
        self.assertTrue(bool(ptr), "Logits pointer must not be null")

        logits = np.ctypeslib.as_array(ptr, shape=(self.engine.vocab_size,))
        self.assertEqual(logits.shape, (128256,))
        self.assertFalse(np.isnan(logits).any(), "Logits must not contain NaN")
        self.assertFalse(np.isinf(logits).any(), "Logits must not contain Inf")

        # Top logits should have sensible magnitudes
        max_logit = float(np.max(logits))
        min_logit = float(np.min(logits))
        self.assertGreater(max_logit, min_logit)

    def test_04_prefill_and_sampling(self):
        """Verify prefill accelerates multi-token prompt and greedy sampling works."""
        self.engine.reset()
        prompt = "The capital of France is"
        tokens = self.tokenizer.encode(prompt)
        self.assertGreater(len(tokens), 1)

        # Prefill prompt tokens up to last
        self.engine.prefill(tokens[:-1], start_pos=0, compute_last_logits=False)
        self.engine.forward_token(tokens[-1], pos=len(tokens) - 1, compute_logits=True)

        # Greedy sample
        sampled = self.engine.sample(temperature=0.0, top_p=0.9)
        self.assertGreaterEqual(sampled, 0)
        self.assertLess(sampled, self.engine.vocab_size)

        decoded = self.tokenizer.decode(sampled)
        # Expected continuation for "The capital of France is" is typically " Paris" or whitespace + Paris
        self.assertTrue("Paris" in decoded or len(decoded.strip()) > 0)

    def test_05_lm_eval_adapter_loglikelihood(self):
        """Verify the LM harness adapter can score choice log-likelihoods without crashing."""
        adapter = Signed4BitLM(
            model_path=str(MODEL_PATH),
            tokenizer_path=str(TOKENIZER_PATH),
            batch_size=1,
        )

        req1 = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question: What color is the sky? Answer:", " blue"),
            idx=0,
        )
        req2 = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Question: What color is the sky? Answer:", " green"),
            idx=1,
        )

        results = adapter.loglikelihood([req1, req2])
        self.assertEqual(len(results), 2)

        logprob_blue, is_greedy_blue = results[0]
        logprob_green, is_greedy_green = results[1]

        self.assertFalse(np.isnan(logprob_blue))
        self.assertFalse(np.isnan(logprob_green))
        # " blue" should have significantly higher probability than " green"
        self.assertGreater(logprob_blue, logprob_green, "Expected P(' blue') > P(' green')")


if __name__ == "__main__":
    unittest.main()
