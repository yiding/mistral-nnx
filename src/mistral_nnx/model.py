"""Mistral inference implemented in Flax NNX.

Can load weights from huggingface model

Conventions used for axis labeling:

- B: batch
- S: seqlen
- V: vocab
- E: embed
- D: head_dim
- H: num_heads
- HQ: num_q_heads
- K: num_kv_heads

Note: The rotary embedding implementation used here is from flaxformers, which
is same as hugginface transformers'. This is not compatible with
mistral-inference's implementation. The order of the features must be swapped if
using mistral weights. More details in class `RotaryEmbedding`.

Not supported:
- Sliding window

"""

import functools
import os
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import flax.core.spmd
import flax.struct
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from flax.typing import Dtype, Initializer, LogicalRules
from jax import Array, ShapeDtypeStruct
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding
from jaxtyping import Bool, Float, Integer
from safetensors import safe_open
from transformers import MistralConfig
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from .embedding import RotaryEmbedding
from .util import keystr_simple, update_sharding

PARAM_INDEX_FILE = "model.safetensors.index.json"


class Causal:
    pass


def tril_mask(
    rows: int,
    cols: int,
    start_index: int,
    dtype=jnp.bool,
):
    """
    Create a triangular mask matrix.

    ```
            v start_index
            0 1 2 3   <- cols
        +--------
        0 | 1 1 0 0
        1 | 1 1 1 0
        2 | 1 1 1 1
        ^ rows
    ```
    """
    return jnp.tri(rows, cols, k=start_index, dtype=dtype)


class Axis(str, Enum):
    EMBED = "embed"
    MLP = "mlp"
    HEAD = "head"
    QHEAD = "qhead"
    KVHEAD = "kvhead"
    VOCAB = "vocab"

    def __str__(self) -> str:
        return self.value


@flax.struct.dataclass
class KVCacheLayer:
    cache_k: Float[Array, "B S H D"]
    cache_v: Float[Array, "B S H D"]
    index: Integer[Array, ""]

    @property
    def max_seqlen(self) -> int:
        return self.cache_k.shape[1]

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: Dtype,
        mesh: Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ) -> "KVCacheLayer":
        assert len(shape) == 4, f"shape should be (B,S,H,D), got: {shape}"

        # KV cache takes the output of K and V projections, which is sharded by
        # KVHEAD, HEAD_DIM.  The KV cache itself should be sharded the same way.
        if sharding_rules is None and mesh is None:
            sharding = SingleDeviceSharding(jax.devices("cpu")[0])
        elif sharding_rules is not None and mesh is not None:
            rules_dict = {k: v for k, v in sharding_rules}
            sharding = NamedSharding(
                mesh,
                PartitionSpec(
                    None, None, rules_dict[Axis.KVHEAD], rules_dict[Axis.HEAD]
                ),
            )
        else:
            raise ValueError("mesh and sharding_rules must be both None or both set")

        return cls(
            cache_k=jnp.zeros(shape, dtype=dtype, device=sharding),
            cache_v=jnp.zeros(shape, dtype=dtype, device=sharding),
            index=jnp.array(0, dtype="uint32"),
        )

    def update(
        self,
        k: Float[Array, "B S H D"],
        v: Float[Array, "B S H D"],
        len: Float[Array, ""],
    ) -> "KVCacheLayer":
        """Update the cache at the given index.

        Can be used for prefill by passing array with seqlen > 1.

        Args:
            k: key array of shape (seqlen, num_kv_heads, head_dim)
            v: value array of same shape
            len: actual length

        Returns:
            updated cache layer with seqlen incremented by the amount from input.
        """
        KB, KS, _KH, _KD = k.shape
        VB, VS, _VH, _VD = v.shape
        assert (KB, KS) == (
            VB,
            VS,
        ), f"k and v should have same batch,seqlen: {(KB, KS)} != {(VB,VS)}"
        Z = jnp.array(0, dtype=self.index.dtype)
        return KVCacheLayer(
            cache_k=jax.lax.dynamic_update_slice(
                self.cache_k, k, (Z, self.index, Z, Z)
            ),
            cache_v=jax.lax.dynamic_update_slice(
                self.cache_v, v, (Z, self.index, Z, Z)
            ),
            index=self.index + len,
        )


@flax.struct.dataclass
class KVCache:
    layers: list[KVCacheLayer]

    @classmethod
    def create(
        cls,
        layers: int,
        batch_size: int,
        max_seqlen: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: Dtype,
        mesh: Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ) -> "KVCache":
        shape = (batch_size, max_seqlen, num_kv_heads, head_dim)
        return cls(
            layers=[
                KVCacheLayer.create(shape, dtype, mesh, sharding_rules)
                for _ in range(layers)
            ]
        )


def _init_with_sharding(
    init_fn: Initializer,
) -> Callable[[tuple[Axis, ...]], Initializer]:
    """Returns a function that when invoked, returns an annotated initializer
    when given logical axis annotations.
    """

    def init(sharding):
        return nnx.with_partitioning(init_fn, sharding=sharding)

    return init


class FeedForward(nnx.Module):

    def __init__(
        self, dim: int, hidden_dim: int, dtype: Any, param_dtype: Dtype, rngs: nnx.Rngs
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.param_dtype = param_dtype

        init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.w1 = nnx.LinearGeneral(
            self.dim,
            self.hidden_dim,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.w2 = nnx.LinearGeneral(
            self.hidden_dim,
            self.dim,
            kernel_init=init((Axis.MLP, Axis.EMBED)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.w3 = nnx.LinearGeneral(
            self.dim,
            self.hidden_dim,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Float[Array, "B S E"]) -> Float[Array, "B S E"]:
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class Attention(nnx.Module):
    """Mistral attention supports different number of Q heads vs KV heads."""

    dim: int
    n_q_heads: int
    head_dim: int
    n_kv_heads: int
    dtype: Any

    wq: nnx.LinearGeneral
    wk: nnx.LinearGeneral
    wv: nnx.LinearGeneral
    wo: nnx.LinearGeneral

    def __init__(
        self,
        dim: int,
        n_q_heads: int,
        head_dim: int,
        n_kv_heads: int,
        rope: RotaryEmbedding,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.n_q_heads = n_q_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.rope = rope
        self.dtype = dtype

        init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.wq = nnx.LinearGeneral(
            self.dim,
            (self.n_q_heads, self.head_dim),
            use_bias=False,
            kernel_init=init((Axis.EMBED, Axis.QHEAD, Axis.HEAD)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wk = nnx.LinearGeneral(
            self.dim,
            (self.n_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, Axis.HEAD)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wv = nnx.LinearGeneral(
            self.dim,
            (self.n_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, Axis.HEAD)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wo = nnx.LinearGeneral(
            (self.n_q_heads, self.head_dim),
            self.dim,
            axis=(-2, -1),
            kernel_init=init((Axis.QHEAD, Axis.HEAD, Axis.EMBED)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @property
    def _queries_per_head(self) -> int:
        return self.n_q_heads // self.n_kv_heads

    def __call__(self, x: Float[Array, "B S E"]) -> Array:
        """
        Args:
          x: Array of shape (batch, seqlen, dim)
        """
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.rope(xq), self.rope(xk)

        out = jax.nn.dot_product_attention(xq, xk, xv, is_causal=True)
        out = self.wo(out)
        return out

    def decode(
        self,
        x: Float[Array, "B S E"],
        cache: KVCacheLayer,
        mask: Bool[Array, "B S S"] | type[Causal] | None = None,
    ) -> tuple[Float[Array, "B S E"], KVCacheLayer]:
        """Decode using KV cache.

        Args:
            x: The next items in the sequence.
            mask: Causal mask to use, see explanation below.

        Returns:
            (output, cache_layer) the output and updated cache layer.


        Regarding `mask`:
        - If array, use a custom causal mask.
        - If `Causal`, use attention implementation's causal mask, which could
          be faster. This is correct only at the start of sequence.
        - If None, use no causal mask. This can be used when predicting a single
            next token.

        """
        B, T, _E = x.shape
        assert B == 1, "only batch size 1 supported for now"

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        start_index = jnp.full((B,), cache.index)

        xq = self.rope(xq, start_index=start_index)
        xk = self.rope(xk, start_index=start_index)

        cache = cache.update(xk, xv, jnp.array(T, dtype=jnp.uint32))
        xk = cache.cache_k
        xv = cache.cache_v

        if mask is Causal:
            actual_mask = None
            is_causal = True
        elif mask is None:
            actual_mask = None
            is_causal = False
        elif isinstance(mask, jax.Array):
            actual_mask = mask
            is_causal = False
        else:
            raise ValueError(f"Invalid mask type: {type(mask)}")

        out = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            mask=actual_mask,
            is_causal=is_causal,
            query_seq_lengths=jnp.array([T], dtype=jnp.int32),
            key_value_seq_lengths=jnp.full((B,), cache.index, dtype=jnp.int32),
        )
        out = self.wo(out)
        return out, cache


class TransformerBlock(nnx.Module):

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        rope: RotaryEmbedding,
        dtype: Any,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
    ):
        init = _init_with_sharding(nnx.initializers.ones_init())

        self.n_q_heads = n_q_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope=rope,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention_norm = nnx.RMSNorm(
            dim,
            epsilon=norm_eps,
            scale_init=init((Axis.EMBED,)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ffn_norm = nnx.RMSNorm(
            dim,
            epsilon=norm_eps,
            scale_init=init((Axis.EMBED,)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "B S E"]) -> Float[Array, "B S E"]:
        r = self.attention(self.attention_norm(x))
        h = x + r
        r = self.mlp(self.ffn_norm(h))
        return h + r

    def decode(
        self,
        x: Float[Array, "B S E"],
        cache: KVCacheLayer,
        mask: Bool[Array, "B S S"] | type[Causal] | None = None,
    ) -> tuple[Float[Array, "B S E"], KVCacheLayer]:
        r, cache = self.attention.decode(self.attention_norm(x), cache, mask)
        h = x + r
        r = self.mlp(self.ffn_norm(h))
        return h + r, cache


class MistralModel(nnx.Module):
    layers: list[TransformerBlock]
    config: MistralConfig
    sharding_rules: LogicalRules | None

    def __init__(
        self,
        config: MistralConfig,
        *,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
        sharding_rules: LogicalRules | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.sharding_rules = sharding_rules  # keep track of sharding rules for creating compatible kv cache

        embed_init = _init_with_sharding(nnx.initializers.lecun_normal())
        norm_init = _init_with_sharding(nnx.initializers.zeros_init())
        linear_init = _init_with_sharding(nnx.initializers.lecun_normal())

        head_dim = config.head_dim
        assert type(head_dim) is int

        rope = RotaryEmbedding(
            features=head_dim,
            length=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.embed = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=embed_init((Axis.VOCAB, Axis.EMBED)),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.norm = nnx.RMSNorm(
            config.hidden_size,
            scale_init=norm_init((Axis.EMBED,)),
            epsilon=config.rms_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=linear_init((Axis.EMBED, Axis.VOCAB)),
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.layers = [
            TransformerBlock(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                n_q_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                norm_eps=config.rms_norm_eps,
                rope=rope,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )
            for _ in range(0, config.num_hidden_layers)
        ]

    def __call__(self, input_ids: Integer[Array, "B S"]) -> Float[Array, "B S V"]:
        """
        Args:
            input_ids: Array of shape (batch, seqlen)
        Returns:
            array of shape (batch, seqlen, vocab_size)
        """
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)
        return logits

    def decode(
        self,
        input_ids: Integer[Array, "B S"],
        cache: KVCache,
        mask: Bool[Array, "B S S"] | type[Causal] | None = None,
    ) -> tuple[Float[Array, "B S V"], KVCache]:
        """Inference mode using kvcache.

        Args:
            input_ids: Array of shape (batch, seqlen)
            cache: the KV Cache, for incremental decoding.
        Returns:
            array of shape (batch, seqlen, vocab_size)
        """
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h, cache.layers[i] = layer.decode(h, cache.layers[i], mask)
        h = self.norm(h)
        logits = self.output(h)
        return logits, cache

    def create_cache(
        self, batch_size: int, max_seqlen: int, mesh: Mesh | None = None
    ) -> KVCache:
        head_dim = self.config.head_dim
        assert type(head_dim) is int
        return KVCache.create(
            len(self.layers),
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            mesh=mesh,
            sharding_rules=self.sharding_rules,
        )

    def save_orbax(self, ckpt_dir: Path, max_file_size: int = 1 * 1024 * 1024 * 1024):
        """Save model parameters with orbax."""
        state = nnx.state(self, nnx.OfType(nnx.Param))
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(ckpt_dir.absolute(), state)
        checkpointer.wait_until_finished()

    @classmethod
    def _load_with(
        cls,
        loader: Callable[[nnx.State], nnx.State],
        config: MistralConfig,
        dtype: Dtype,
        param_dtype: Dtype,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ):
        """Create a model instance using the specific weight loading function.

        Args:
            loader: weight loading function that takes abstract Param state and
                returns loaded Params.
            mesh: if specified with sharding rules, load to assigned devices.
                Otherwise, load to `SingleDeviceSharding(jax.devices("cpu")[0])`.
        """
        abs_model = nnx.eval_shape(
            lambda: cls(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=nnx.Rngs(0),
                sharding_rules=sharding_rules,
            )
        )
        graphdef = nnx.graphdef(abs_model)
        abs_params = nnx.state(abs_model, nnx.OfType(nnx.Param))

        # annotate abstract params with sharding
        if sharding_rules is not None and mesh is not None:
            # this is a bit awkward, can probably be done within eval_shape.
            with flax.core.spmd.logical_axis_rules(sharding_rules):
                pspecs = nnx.get_partition_spec(abs_params)

            def add_sharding(param: ShapeDtypeStruct, p):
                return update_sharding(param, jax.sharding.NamedSharding(mesh, p))

            abs_params = jax.tree.map(add_sharding, abs_params, pspecs)
        elif sharding_rules is None and mesh is None:
            single = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
            abs_params = jax.tree.map(lambda x: update_sharding(x, single), abs_params)
        else:
            raise ValueError(
                "sharding_rules and mesh should both be specified or both None"
            )

        # Initialize non-param states from an actual new instance.
        # The non-returned arrays should be eliminated by jit.
        # This gets things like precomputed rope-embedding constants.
        @jax.jit
        def non_param():
            model = cls(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=nnx.Rngs(0),
                sharding_rules=sharding_rules,
            )
            return nnx.state(model, nnx.Not(nnx.OfType(nnx.Param)))

        non_params = jax.block_until_ready(non_param())
        params = loader(abs_params)

        return nnx.merge(graphdef, non_params, params)

    @classmethod
    def load(
        cls,
        model_dir: Path,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: Sequence[tuple[str, str]] | None = None,
    ) -> "MistralModel":
        """Load converted hf model.

        Args:
            model_dir: path to the pre-converted model.
            dtype: computation dtype
            param_dtype: dtype of the parameters
            mesh: mesh used for sharding, should be set with sharding_rules.
            sharding_rules:
                If set, load the weights with the correct sharding, otherwise,
                weights are loaded as unsharded single device (jax default).
        """
        config = MistralConfig.from_pretrained(model_dir)
        assert isinstance(config, MistralConfig)
        return cls._load_with(
            functools.partial(_load_orbax, model_dir / "orbax"),
            config,
            dtype,
            param_dtype,
            mesh,
            sharding_rules,
        )

    @classmethod
    def load_from_hf_pt_model(
        cls,
        model_name: str,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: Sequence[tuple[str, str]] | None = None,
    ) -> "MistralModel":
        """Load model from HF pytorch model.

        Renames, transposes, and reshapes tensors as necessary.

        Args:
            model_name: HF model name
            dtype: computation dtype
            param_dtype: dtype of the parameters
            sharding_rules:
                If set, load the weights with the correct sharding, otherwise,
                weights are loaded as unsharded single device (jax default).

        Returns:
            MistralModel with loaded weights.
        """
        config = MistralConfig.from_pretrained(model_name)
        assert isinstance(config, MistralConfig)

        return cls._load_with(
            functools.partial(_load_hf_pt_params, model_name),
            config,
            dtype,
            param_dtype,
            mesh,
            sharding_rules,
        )


def _load_orbax(ckpt_dir: Path, abs_state: nnx.State) -> nnx.State:
    """Load model from orbax checkpoint."""
    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(ckpt_dir.absolute(), abs_state)


def _load_hf_pt_params(hf_model: str, abs_state: nnx.State) -> nnx.State:
    """Load model from HF pytorch model.

    Renames, transposes, and reshapes tensors as necessary.

    Args:
        hf_model: HF model name or path.
        abs_state: abstract state of nnx.Params of the model.

    Returns:
        nnx.State with actual param arrays that can be `nnx.merge`d into the
        model.
    """
    PARAM_INDEX_FILE = "model.safetensors.index.json"
    index = cached_file(hf_model, PARAM_INDEX_FILE)
    shard_paths, meta = get_checkpoint_shard_files(hf_model, index)
    assert isinstance(shard_paths, list)

    with ExitStack() as stack:
        shards = {
            os.path.basename(s): stack.enter_context(safe_open(s, "flax"))
            for s in shard_paths
        }

        def transpose_only(param: jax.Array, _abs_param):
            return param.transpose()

        def transpose_reshape(param: jax.Array, abs_param):
            return param.transpose().reshape(abs_param.shape)

        def identity(param: jax.Array, _abs_param):
            return param

        def load_one(path, abs_param):
            # Map to hf pt model (name, need_transpose)
            name_map = {
                "embed/embedding/value": ("model.embed_tokens.weight", identity),
                "output/kernel/value": ("lm_head.weight", transpose_only),
                "norm/scale/value": ("model.norm.weight", identity),
            }
            layer_name_map = {
                "attention/wq/kernel/value": (
                    "self_attn.q_proj.weight",
                    transpose_reshape,
                ),
                "attention/wk/kernel/value": (
                    "self_attn.k_proj.weight",
                    transpose_reshape,
                ),
                "attention/wv/kernel/value": (
                    "self_attn.v_proj.weight",
                    transpose_reshape,
                ),
                "attention/wo/kernel/value": (
                    "self_attn.o_proj.weight",
                    transpose_reshape,
                ),
                "attention_norm/scale/value": ("input_layernorm.weight", identity),
                "mlp/w1/kernel/value": ("mlp.gate_proj.weight", transpose_only),
                "mlp/w2/kernel/value": ("mlp.down_proj.weight", transpose_only),
                "mlp/w3/kernel/value": ("mlp.up_proj.weight", transpose_only),
                "ffn_norm/scale/value": (
                    "post_attention_layernorm.weight",
                    identity,
                ),
            }
            if path[0].key == "layers":
                idx = path[1].key
                layer_name, postprocess = layer_name_map[
                    keystr_simple(path[2:], separator="/")
                ]
                name = f"model.layers.{idx}.{layer_name}"
            else:
                name, postprocess = name_map[keystr_simple(path, separator="/")]

            param = shards[meta["weight_map"][name]].get_tensor(name)
            param = postprocess(param, abs_param)
            assert (
                param.shape == abs_param.shape
            ), f"Wrong shape for {keystr_simple(path, separator='/')}. Expected: {abs_param.shape}, actual: {param.shape}"
            sharding = abs_param.sharding
            assert isinstance(sharding, jax.sharding.Sharding)
            return jax.device_put(param.astype(abs_param.dtype), device=sharding)

        return jax.tree_util.tree_map_with_path(load_one, abs_state)


def convert_hf_model(hf_model: str, output_path: Path, param_dtype=jnp.bfloat16):
    import shutil

    from transformers import AutoTokenizer

    model = MistralModel.load_from_hf_pt_model(hf_model, param_dtype=param_dtype)

    output_path.mkdir(exist_ok=True)
    checkpoint_dir: Path = (output_path / "orbax").absolute()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    model.save_orbax(checkpoint_dir)

    # Config and tokenizer
    model.config.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(hf_model).save_pretrained(output_path)
