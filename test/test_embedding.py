import jax.numpy as jnp

from mistral_nnx.embedding import RotaryEmbedding


def test_index():
    # When using GPU the values may differ by 1e-3... why?
    B, S, Q, K, D = (1, 5, 4, 2, 16)

    rope = RotaryEmbedding(features=D, length=32768, theta=100000000.0)

    q = jnp.ones((B, S, Q, D), dtype=jnp.float32)

    # Check `index` parameter gives same result for a single `q` as a long
    # sequence.
    eq = rope(q)
    for i in range(S):
        q2 = rope(q[:, [i], :, :], start_index=jnp.array([i], dtype=jnp.int32))
        assert jnp.allclose(eq[:, [i], :, :], q2)
