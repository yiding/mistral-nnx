#!/usr/bin/env python3
import collections
import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import mistral_nnx
from mistral_nnx import generate

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


if __name__ == "__main__":
    unittest.main()
