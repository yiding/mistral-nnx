import collections

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import mistral_nnx
import mistral_nnx.generate as generate


def test_generate():
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
    generator = generate.Generator(model, max_seqlen=256)
    r = generator.generate([1, 2, 3, 4, 5], rngs=nnx.Rngs(0), max_tokens=15)
    assert len(r.logits) > 5


def test__sample_top_p():
    probs = jnp.array([[0, 0.3, 0.6], [0.6, 0.2, 0]], dtype=jnp.float32)
    key = jax.random.key(0)
    indexes = generate._sample_top_p(probs, 0.5, key)
    expected = jnp.array([2, 0], dtype="int32")

    assert (expected == indexes).all()


def test_sample_best():
    probs = jnp.array([[0, 0.3, 0.6], [0.6, 0.2, 0]], dtype=jnp.float32)
    indexes = generate.sample_best(probs)
    expected = jnp.array([2, 0], dtype="int32")
    assert (expected == indexes).all()


def test_sample_top_p():
    # small top_p value selects only the most likely.
    probs = jnp.array([[0.1, 0.2, 0.7, 0.8], [0.8, 0.7, 0.2, 0.1]], dtype=jnp.float32)
    for seed in range(0, 1000):
        key = jax.random.key(seed)
        # Use a small top_p to ensure only the most likely gets selected.
        expected = jnp.array([3, 0], dtype="int32")
        actual = generate.sample_top_p(probs, temperature=1.0, top_p=0.2, key=key)
        assert (
            expected == actual
        ).all(), f"should be equal. expected={expected} actual={actual} seed={seed}"

    # larger top-p selects the top 2 most likely.
    probs = jnp.array([0.8, 0.7, 0.2, 0.1], dtype=jnp.float32)
    counts = collections.Counter()
    for seed in range(0, 1000):
        key = jax.random.key(seed)
        # Use a small top_p to ensure only the most likely gets selected.
        chosen = generate.sample_top_p(probs, temperature=0.7, top_p=0.5, key=key)
        counts[chosen.item()] += 1
    assert counts[0] > counts[1], "most likely option is chosen more frequently"
    assert list(sorted(counts.keys())) == [
        0,
        1,
    ], "only most likely options have been chosen"
