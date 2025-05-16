from jaxtyping import Float, Integer
from jax import Array
import flax.nnx as nnx
import jax.numpy as jnp
import jax


def sample_best(logits: Float[Array, "*B V"]) -> Integer[Array, "*B"]:
    return jnp.argmax(logits, axis=-1)


def sample_top_p(
        logits: Float[Array, "*B V"], 
        *, 
        temperature: float,
        top_p: float,
        key: Array,
) -> Integer[Array, "*B"]:
    """Top-p sampling.
    
    Args:
        key: RNG key
    """
    probs = nnx.softmax(logits / temperature, axis=-1)
    return _sample_top_p(probs, top_p, key=key)


def _sample_top_p(
    probs: Float[Array, "*B V"], p: float, key: Array
) -> Float[Array, "*B"]:
    """Sample a token using top-p sampling."""
    # From flax/examples/gemma/sampler.py
    probs_sorted, indices = jax.lax.top_k(probs, k=probs.shape[-1])
    cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
    mask = cumsum_probs - probs_sorted > p
    probs_sorted = jnp.where(mask, 0.0, probs_sorted)
    probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

    next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

    next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
    next_token = jnp.squeeze(next_token, axis=-1)
    return next_token
