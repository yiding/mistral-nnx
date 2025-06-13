#!/usr/bin/env python3
import collections

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import mistral_nnx
from mistral_nnx.model import Causal

# on cuda using float16/float32 results in weird precision errors causing test
# failures. bfloat16 seems to work fine everywhere.
DTYPE = jnp.bfloat16
SEQLEN = 10
EMBED = 32
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16


@pytest.fixture
def attention_layer():
    return mistral_nnx.Attention(
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


@pytest.fixture
def cache_layer() -> mistral_nnx.KVCacheLayer:
    return mistral_nnx.KVCacheLayer.create((1, SEQLEN, KV_HEADS, HEAD_DIM), dtype=DTYPE)


def test_kvcache_update(cache_layer):
    """Check kv cache handles writing seqlen>1 correctly."""
    assert cache_layer.index.item() == 0

    cache_layer = cache_layer.update(
        jnp.ones((1, 1, KV_HEADS, HEAD_DIM), dtype=DTYPE),
        jnp.ones((1, 1, KV_HEADS, HEAD_DIM), dtype=DTYPE),
        jnp.array(1, dtype=jnp.uint32),
    )
    assert cache_layer.index.item() == 1
    assert (cache_layer.cache_k[0, 0, ...] == 1).all()
    assert (cache_layer.cache_v[0, 0, ...] == 1).all()

    cache_layer = cache_layer.update(
        jnp.full((1, 3, KV_HEADS, HEAD_DIM), 2, dtype=DTYPE),
        jnp.full((1, 3, KV_HEADS, HEAD_DIM), 2, dtype=DTYPE),
        jnp.array(2, dtype=jnp.uint32),
    )
    assert cache_layer.index.item() == 3
    assert (cache_layer.cache_k[0, 0, ...] == 1).all()
    assert (cache_layer.cache_v[0, 0, ...] == 1).all()
    assert (cache_layer.cache_k[0, 1:3, ...] == 2).all()
    assert (cache_layer.cache_v[0, 1:3, ...] == 2).all()


def test_attention_decode_fill_multi(attention_layer, cache_layer):
    """Check attention decode can prefill kvcache with multiple tokens at a
    time.
    """
    key = jax.random.key(0)
    inputs = jax.random.uniform(key, (1, SEQLEN, EMBED), dtype=DTYPE)

    # decode the first 3 tokens
    out1, cache_layer = attention_layer.decode(
        inputs[:, 0:3, ...], cache_layer, mask=Causal
    )
    assert cache_layer.index.item() == 3

    # and the rest..
    out2, cache_layer = attention_layer.decode(
        inputs[:, 3:, ...],
        cache_layer,
        mask=mistral_nnx.tril_mask(SEQLEN - 3, SEQLEN, 3),
    )
    assert cache_layer.index.item() == SEQLEN

    # and this should equal doing it all together.
    expected = attention_layer(inputs)
    actual = jnp.concatenate([out1, out2], axis=1)

    assert jnp.allclose(expected, actual), "decoding with kvcache should match batched"


def test_decode_same_as_call(attention_layer, cache_layer):
    # Check kv decode produces same result as calling the module on the full
    # sequence.
    key = jax.random.key(0)
    inputs = jax.random.uniform(key, (1, SEQLEN, EMBED), dtype=DTYPE)

    expected = attention_layer(inputs)

    for i in range(0, SEQLEN):
        out, cache_layer = attention_layer.decode(inputs[:, [i], :], cache_layer)
        assert jnp.allclose(
            expected[:, [i], :],
            out,
        ), f"Decode with kv cache should match batched, item={i}"
