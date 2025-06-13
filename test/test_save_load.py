import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_flatten_with_path

from mistral_nnx import MistralConfig, MistralModel


def compare_pytrees(left, right, eps_abs = 1e-6):
    lleaves, ldef = tree_flatten_with_path(left)
    rleaves, rdef = tree_flatten_with_path(right)
    assert ldef == rdef, "tree structures should be the same"

    for ((lpath, l), (rpath, r)) in zip(lleaves, rleaves):
        assert lpath == rpath, "should be checking corresponding elements"
        max_abs_diff = jnp.abs(l - r).max()
        assert max_abs_diff < eps_abs, f"tree leaves should be the same at {lpath}"


@pytest.fixture
@nnx.jit
def small_model() -> MistralModel:
    # Running this under jit since normal load flow runs under jit, this
    # prevents numerical precision differences.
    return MistralModel(
        MistralConfig(
            head_dim=32,
            vocab_size=131072,
            max_position_embeddings=1024,
            intermediate_size=128,
            hidden_size=32,
            num_attention_heads=16,
            num_hidden_layers=4,
            num_key_value_heads=8,
        ),
        param_dtype=jnp.bfloat16,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )


def test_save_load(tmp_path, small_model: MistralModel):
    model_path = tmp_path / "model"
    small_model.config.save_pretrained(model_path)
    weights_path = model_path / "orbax"
    small_model.save_orbax(weights_path)

    loaded_model = MistralModel.load(model_path)
    compare_pytrees(nnx.state(small_model), nnx.state(loaded_model))
