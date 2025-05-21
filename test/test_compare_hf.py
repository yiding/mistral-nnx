import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import sys
import mistral_nnx
from mistral_nnx.util import timer
import transformers

import jax
import jax.numpy as jnp
import flax.nnx as nnx

MESH = jax.make_mesh((4,), axis_names=('x',))
SHARDING_RULES = list({
    mistral_nnx.Axis.EMBED: None,
    mistral_nnx.Axis.MLP: "x",
    mistral_nnx.Axis.HEAD: "x",
    mistral_nnx.Axis.QHEAD: None,
    mistral_nnx.Axis.KVHEAD: None,
    mistral_nnx.Axis.VOCAB: None,
}.items())


MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
WEIGHTS = "./Mistral-Small-24B-Instruct-2501.bfloat16.nnx"
INPUT = (
    "[INST]What is the name of the largest planet in our solar system?[/INST] Jupiter"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)


def get_hf_result():
    with timer("HF model loading"):
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL)
    tokens = tokenizer(INPUT, return_tensors="pt")
    with timer("HF model inference"):
        return jnp.array(hf_model.forward(**tokens).logits.detach())


def get_nnx_result():
    with timer("NNX model weight conversion"):
        mistral_nnx.convert_and_save_if_not_exist(
            WEIGHTS, MODEL, param_dtype=jnp.bfloat16
        )
    
    with timer("NNX model loading"):
        nnx_model = mistral_nnx.MistralModel.load(
            MODEL, WEIGHTS, dtype=jnp.float32, sharding_rules=SHARDING_RULES)

    @jax.jit
    def jit_model(graphdef, state, input):
        model = nnx.merge(graphdef, state)
        return model(input)

    graphdef, state = nnx.split(nnx_model)
    tokens = tokenizer(INPUT, return_tensors="jax")

    with timer("NNX jit compile"):
        compiled_model = jit_model.lower(graphdef, state, tokens["input_ids"]).compile()

    with timer("NNX model inference"):
        result_nnx = compiled_model(graphdef, state, tokens["input_ids"])
    return result_nnx


with MESH:
    nnx_result = get_nnx_result()
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
assert jnp.all(hf_argmax == nnx_argmax), "expected argmax to match."
