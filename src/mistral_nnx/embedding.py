# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rotary embedding implementation.

This implementation expects the features to be ordered in odds and evens.

Based on flaxformers/components/embedding.py
"""

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Integer


class RotaryEmbedding(nnx.Module):
    def __init__(self, features: int, length: int, theta: float):
        """
        Args:
            features: Number of features
            length: Max context length
            theta: RoPE theta term
        """
        sin, cos = generate_fixed_pos_embedding(features, length, max_timescale=theta)
        self.sin = nnx.Variable(sin)
        self.cos = nnx.Variable(cos)

    def __call__(
        self,
        arr: Float[Array, "B S H D"],
        start_index: Integer[Array, "B"] | None = None,
    ) -> Float[Array, "B S H D"]:
        """Apply rotary embedding to a query or key array.

        Args:
            arr: query of shape (batch, seqlen, heads, head_dim)
            start_index: start index in the sequence for each batch.

        Returns:
            array with embedding applied.

        Note: This expects features to be ordered in odds and evens, i.e.
          `x1, x3, x5 ... x2, x4, x6 ...`.

          The weights for the whole model should be in this order. If using
          weights that expect features to be (i.e. the mistral weights as used by
          mistral-inference lib), it has to be converted into this order.
        """
        if start_index is None:
            start_index = jnp.zeros((arr.shape[0],), dtype=jnp.uint32)
        return jax.vmap(
            _apply_rotary_embedding,
            in_axes=(0, 0, None, None),
            out_axes=0,
        )(arr, start_index, self.cos.value, self.sin.value)


def rotate_half(x):
    """Helper that splits a tensor at last dim into half and rotate it."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


def _apply_rotary_embedding(
    arr: Float[Array, "S H D"],
    start_index: Integer[Array, ""],
    cos: Float[Array, "L D"],
    sin: Float[Array, "L D"],
):
    S, H, D = arr.shape
    zero = jnp.array(0, dtype=start_index.dtype)

    # Get the portion of the embedding vector starting at the index.
    cos_slice = jax.lax.dynamic_slice(cos, (start_index, zero), (S, D))
    sin_slice = jax.lax.dynamic_slice(sin, (start_index, zero), (S, D))

    # Replicate for each head.
    cos_slice = jax.lax.broadcast_in_dim(cos_slice, (S, H, D), (0, 2))
    sin_slice = jax.lax.broadcast_in_dim(sin_slice, (S, H, D), (0, 2))

    rotated = arr * cos_slice + rotate_half(arr) * sin_slice
    return rotated.astype(arr.dtype)


def generate_fixed_pos_embedding(
    features, length, min_timescale=1.0, max_timescale=10000.0
):
    """Generate Sin/Cos for Rotary Embeddings.

    Generates sinusoids at (features//2) different timescales, where the
    timescales form a gemetric series from min_timescale to max_timescale
    (max_timescale is not included, but would be the next element in the series).

    Sinusoids are evaluated at integer positions i in [0, length).

    The outputs are computed as:

      output_sin[i, j] = sin(i / timescale[j])
      output_cos[i, j] = cos(i / timescale[j])

    Finally, the outputs are tiled twice in the features dimension.

    Args:
      features: an integer
      length: an integer
      min_timescale: an optional float
      max_timescale: an optional float

    Returns:
      output_sin: a float32 Tensor with shape [length, features]
      output_cos: a float32 Tensor with shape [length, features]
    """
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, since rounding off to a bfloat16 is
    # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
    # from sin(256).
    sinusoid_inp = jnp.einsum(
        "i , j -> i j",
        jnp.arange(length, dtype=jnp.float32),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)

    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
