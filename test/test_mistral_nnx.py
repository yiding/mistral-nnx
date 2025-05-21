#!/usr/bin/env python3
import mistral_nnx
from mistral_nnx import sampler

import collections
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import unittest

# on cuda using float16/float32 results in weird precision errors causing test
# failures. bfloat16 seems to work fine everywhere.
DTYPE = jnp.bfloat16


class TestCase(unittest.TestCase):
    def assertAllClose(self, a, b, msg=None):
        if not jnp.allclose(a, b):
            msg = f"{msg or 'Arrays are not close.'}\na =\n{a}\nb =\n{b}"
            self.fail(msg=msg)

    def assertAllEqual(self, a, b, msg=None):
        if not (a == b).all():
            self.fail(msg=msg or f"Array elements are not equal.\na =\n{a}\nb =\n{b}")


class TestSample(TestCase):
    def test__sample_top_p(self):
        probs = jnp.array([[0, 0.3, 0.6], [0.6, 0.2, 0]], dtype=DTYPE)
        key = jax.random.key(0)
        indexes = sampler._sample_top_p(probs, 0.5, key)
        expected = jnp.array([2, 0], dtype="int32")
        self.assertAllEqual(expected, indexes)

    def test_sample_best(self):
        probs = jnp.array([[0, 0.3, 0.6], [0.6, 0.2, 0]], dtype=DTYPE)
        indexes = sampler.sample_best(probs)
        expected = jnp.array([2, 0], dtype="int32")
        self.assertAllEqual(expected, indexes)

    def test_sample_top_p(self):
        # small top_p value selects only the most likely.
        probs = jnp.array([[0.1, 0.2, 0.7, 0.8], [0.8, 0.7, 0.2, 0.1]], dtype=DTYPE)
        for seed in range(0, 1000):
            key = jax.random.key(seed)
            # Use a small top_p to ensure only the most likely gets selected.
            expected = jnp.array([3, 0], dtype="int32")
            actual = sampler.sample_top_p(probs, temperature=1.0, top_p=0.2, key=key)
            self.assertAllEqual(
                expected,
                actual,
                f"should be equal. expected={expected} actual={actual} seed={seed}",
            )

        # larger top-p selects the top 2 most likely.
        probs = jnp.array([0.8, 0.7, 0.2, 0.1], dtype=DTYPE)
        counts = collections.Counter()
        for seed in range(0, 1000):
            key = jax.random.key(seed)
            # Use a small top_p to ensure only the most likely gets selected.
            chosen = sampler.sample_top_p(probs, temperature=0.7, top_p=0.5, key=key)
            counts[chosen.item()] += 1
        self.assertGreater(
            counts[0], counts[1], "most likely option is chosen more frequently"
        )
        self.assertEqual(
            list(sorted(counts.keys())), [0, 1], "only most likely options have been chosen"
        )


class TestRotaryEmbedding(TestCase):
    def test_index(self):
        # When using GPU the values may differ by 1e-3... why?
        B, S, Q, K, D = (1, 5, 4, 2, 16)

        rope = mistral_nnx.RotaryEmbedding(features=D, length=32768, theta=100000000.0)

        q = jnp.ones((B, S, Q, D), dtype=DTYPE)
        k = jnp.ones((B, S, K, D), dtype=DTYPE)

        # Check `index` parameter gives same result for a single `q` as a long
        # sequence.
        eq, ek = rope(q, k)
        for i in range(S):
            q2, k2 = rope(q[:, [i], :, :], k, index=jnp.array([i], dtype=jnp.int32))
            self.assertAllClose(eq[:, [i], :, :], q2)
            self.assertAllClose(ek, k2)


class TestAttention(TestCase):
    def test_decode_same_as_call(self):
        # Check kv decode produces same result as calling the module on the full
        # sequence.
        SEQLEN = 10
        EMBED = 32
        Q_HEADS = 4
        KV_HEADS = 2
        HEAD_DIM = 16

        key = jax.random.key(0)
        inputs = jax.random.uniform(key, (1, SEQLEN, EMBED), dtype=DTYPE)
        module = mistral_nnx.Attention(
            dim=EMBED,
            n_q_heads=Q_HEADS,
            head_dim=HEAD_DIM,
            n_kv_heads=KV_HEADS,
            rope=mistral_nnx.RotaryEmbedding(
                features=HEAD_DIM, length=32768, theta=100000000.0
            ),
            dtype=DTYPE,
            param_dtype=DTYPE,
            rngs=nnx.Rngs(0),
        )

        expected = module(inputs)

        cache = mistral_nnx.KVCache.create(
            1, 1, SEQLEN, KV_HEADS, HEAD_DIM, dtype=DTYPE
        ).layers[0]

        for i in range(0, SEQLEN):
            out, cache = module.decode(inputs[:, [i], :], cache)
            self.assertAllClose(
                expected[:, [i], :],
                out,
                msg=f"Decode with kv cache should match batched, item={i}",
            )


class TestGenerate(TestCase):
    def test_generate(self):
        # Test all the plumbing is working.
        config = mistral_nnx.MistralConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        rngs = nnx.Rngs(params=0)
        model = mistral_nnx.MistralModel(
            config,
            dtype="float32",
            param_dtype="float32",
            rngs=rngs,
        )
        generator = mistral_nnx.Generator(model, max_tokens=256)
        r = generator.generate([1, 2, 3, 4, 5], rngs=nnx.Rngs(0), max_tokens=15)
        self.assertTrue(len(r) > 5)


if __name__ == "__main__":
    unittest.main()
