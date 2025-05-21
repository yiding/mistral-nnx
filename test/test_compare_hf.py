import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import sys
import mistral_nnx
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
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL)
    tokens = tokenizer(INPUT, return_tensors="pt")
    return jnp.array(hf_model.forward(**tokens).logits.detach())


def get_nnx_result():
    weights = mistral_nnx.convert_and_save_if_not_exist(
        WEIGHTS, MODEL, param_dtype=jnp.bfloat16
    )
    nnx_model = mistral_nnx.load_model(
        MODEL, WEIGHTS, dtype=jnp.float32, sharding_rules=SHARDING_RULES)

    @jax.jit
    def jit_model(graphdef, state, input):
        model = nnx.merge(graphdef, state)
        return model(input)

    graphdef, state = nnx.split(nnx_model)

    tokens = tokenizer(INPUT, return_tensors="jax")
    result_nnx = jit_model(graphdef, state, tokens["input_ids"])
    return result_nnx


with MESH:
    nnx_result = get_nnx_result()
hf_result = get_hf_result()
diff = hf_result - nnx_result
print(f"Max diff: {diff.max()}")
print(f"Min diff: {diff.min()}")
print(f"Mean diff: {diff.mean()}")

if jnp.abs(diff).max() < 1e-4:
    print("PASS expected max(|diff|) < 1e-4")
    sys.exit(0)
else:
    print("FAIL expected max(|diff|) < 1e-4")
    sys.exit(1)


if __name__ == "__main__":
    main()