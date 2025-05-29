"""Tests the model implementation itself.

Can be run with multiple devices, e.g. via

    XLA_FLAGS=--xla_force_host_platform_device_count=4
"""

from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
import transformers
from transformers import PreTrainedTokenizer

import mistral_nnx
import mistral_nnx.generate
from mistral_nnx.util import timer

MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
NNX_MODEL = Path("./Mistral-Small-24B-Instruct-2501-NNX")

SHARDING_RULES = list(
    {
        mistral_nnx.Axis.EMBED: None,
        mistral_nnx.Axis.MLP: "x",
        mistral_nnx.Axis.HEAD: "x",
        mistral_nnx.Axis.QHEAD: None,
        mistral_nnx.Axis.KVHEAD: None,
        mistral_nnx.Axis.VOCAB: None,
    }.items()
)


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module", autouse=True)
def mesh():
    # Global mesh is set for all tests in this module, so no explicit 'with
    # mesh: ' is needed.  Still a good idea to declare the fixture in the test
    # params for clarity.
    devices = jax.devices("cpu")
    mesh = jax.make_mesh((len(devices),), axis_names=("x",), devices=devices)
    with mesh:
        yield mesh


@pytest.fixture(scope="module")
def nnx_model(mesh: jax.sharding.Mesh) -> nnx.Module:
    with timer("NNX model weight conversion"):
        if not NNX_MODEL.exists():
            mistral_nnx.convert_hf_model(MODEL, NNX_MODEL)

    with timer("NNX model loading"):
        return mistral_nnx.MistralModel.load(
            NNX_MODEL, dtype=jnp.float32, mesh=mesh, sharding_rules=SHARDING_RULES,
        )


def load_hf_model() -> transformers.MistralForCausalLM:
    with timer("HF model loading"):
        return transformers.AutoModelForCausalLM.from_pretrained(MODEL)


def test_compare_hf(tokenizer, mesh, nnx_model):
    """Compare model implementation output vs huggingface torch model.

    HF model is loaded and discarded to reduce memory usage.
    """
    input = "[INST]What is the name of the largest planet in our solar system?[/INST] Jupiter"

    def get_nnx_result():
        @jax.jit
        def jit_model(graphdef, state, input):
            model = nnx.merge(graphdef, state)
            return model(input)

        graphdef, state = nnx.split(nnx_model)
        tokens = tokenizer(input, return_tensors="jax")["input_ids"]

        with timer("test_compare_hf - jit compile"):
            compiled_model = jit_model.lower(graphdef, state, tokens).compile()

        with timer("test_compare_hf - nnx forward pass"):
            nnx_result = compiled_model(graphdef, state, tokens)
            nnx_result.block_until_ready()
            return nnx_result

    nnx_result = get_nnx_result()

    def get_hf_result():
        tokens = tokenizer(input, return_tensors="pt")
        with timer("test_compare_hf - hf forward pass"):
            logits = load_hf_model().forward(**tokens).logits
            assert logits is not None
            return jnp.array(logits.detach())
    hf_result = get_hf_result()

    # Check that the results are close enough.
    diff = hf_result - nnx_result
    abs_max_diff = jnp.abs(diff).max()
    print(f"max(|diff|) = {abs_max_diff}")
    assert abs_max_diff < 1e-4, f"expected max(|diff|) < 1e-4"

    # Check argmax are the same
    hf_argmax = jnp.argmax(hf_result, axis=-1)
    nnx_argmax = jnp.argmax(nnx_result, axis=-1)
    print(f"hf_argmax  = {hf_argmax}")
    print(f"nnx_argmax = {nnx_argmax}")
    assert hf_argmax.tolist() == nnx_argmax.tolist(), "expected argmax to match."


def test_generate(tokenizer, mesh, nnx_model):
    """Compare output using the kv-cache decoding `Generator` to the simple forward pass."""
    input = "[INST]31 * 12 = [/INST] "
    generator = mistral_nnx.generate.Generator(nnx_model, max_seqlen=30)
    tokens = tokenizer(input)["input_ids"]
    rngs = nnx.Rngs(0)

    with timer("test_generate - Decoding w/ kv cache"):
        result = generator.generate(tokens, rngs=rngs, max_tokens=10, mesh=mesh)

    # run tokens through forward pass
    all_tokens = jnp.array(result.tokens)[None, ...]

    @jax.jit
    def jit_model(graphdef, state, input):
        model = nnx.merge(graphdef, state)
        return model(input)

    graphdef, state = nnx.split(nnx_model)

    with timer("test_generate - jit compile forward pass"):
        compiled_model = jit_model.lower(graphdef, state, all_tokens).compile()

    with timer("test_generate - run forward pass"):
        all_logits = compiled_model(graphdef, state, all_tokens)
        all_logits.block_until_ready()

    assert jnp.allclose(
        all_logits, result.logits[None, ...], atol=1e-4
    ), "expected kv-cache generated logits to match forward pass."
