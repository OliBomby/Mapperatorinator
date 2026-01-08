"""PyTorch VarWhisper model."""
import copy
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GradientCheckpointingLayer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)

from .configuration_varwhisper import VarWhisperConfig

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func, \
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object


logger = logging.get_logger(__name__)


def _unpad_varwhisper_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def _unpad_encoder_hidden_states(encoder_hidden_states: torch.Tensor,):
    # encoder_hidden_states contains no padding, so we just flatten it and compute cu_seqlens
    batch, seqlen, *rest = encoder_hidden_states.shape
    shape = batch * seqlen
    unpadded_encoder_hidden_states = encoder_hidden_states.view(shape, *rest)

    encoder_cu_seqlens = torch.arange(0, shape + 1, step=seqlen, dtype=torch.int32, device=encoder_hidden_states.device)

    return unpadded_encoder_hidden_states, encoder_cu_seqlens, seqlen


def _pad_varwhisper_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        qkv: (total_nnz, 3, nheads, headdim) - input tensor for packed QKV.
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class VarWhisperFlashRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings applied directly to unpadded sequences.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
            up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
            the cos_sin_cache will be recomputed during the forward pass.
        """
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply rotary embedding *inplace* to qkv.
        qkv: (total_nnz, 3, nheads, headdim)
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen: int max seq length in the batch
        """
        if cu_seqlens is None:
            return super().forward(qkv, max_seqlen=max_seqlen)

        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class VarWhisperRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(
            self,
            config: VarWhisperConfig,
            max_seqlen: Optional[int] = None,
            device=None
    ):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = max_seqlen
        self.original_max_seq_len = max_seqlen

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=max_seqlen)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: "VarWhisperAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    module: "VarWhisperAttention",
    qkv: torch.Tensor,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    rotary_emb: Optional[VarWhisperFlashRotaryEmbedding] = None,
    kv: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    **_kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    if rotary_emb is not None and kv is None:
        qkv = rotary_emb.forward(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    orig_dtype = qkv.dtype

    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        qkv = qkv.to(target_dtype)
        if kv is not None:
            kv = kv.to(target_dtype)

    if kv is None:
        if cu_seqlens is None:
            attn_func = partial(flash_attn_qkvpacked_func, qkv=qkv)
        else:
            attn_func = partial(flash_attn_varlen_qkvpacked_func, qkv=qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    else:
        if cu_seqlens is None:
            attn_func = partial(flash_attn_kvpacked_func,q=qkv, kv=kv)
        else:
            attn_func = partial(flash_attn_varlen_kvpacked_func, q=qkv, kv=kv, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen_k)

    attn = attn_func(
        dropout_p=module.attention_dropout if module.training else 0.0,
        deterministic=module.deterministic_flash_attn,
        window_size=local_attention,
        causal=module.is_causal
    )

    if convert_dtype:
        attn = attn.to(orig_dtype)  # type: ignore

    if cu_seqlens is None:
        return attn.view(bs, -1, dim), None, None

    return attn.view(bs, dim), None, None


def sdpa_attention_forward(
    module: "VarWhisperAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    **_kwargs,
) -> tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


VARWHISPER_ATTENTION_FUNCTION = {
    # (implementation, type)
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class VarWhisperAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(
            self,
            config: VarWhisperConfig,
            num_heads: int,
            max_position_embeddings: int,
            layer_id: int = 1,
            is_causal: bool = False,
            is_cross_attention: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        if config.d_model % self.num_heads != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention heads ({self.num_heads})"
            )

        self.attention_dropout: float = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.head_dim = config.d_model // self.num_heads
        self.all_head_size = self.head_dim * self.num_heads
        if self.is_cross_attention:
            self.Wq = nn.Linear(config.d_model, 1 * self.all_head_size, bias=config.attention_bias)
            self.Wkv = nn.Linear(config.d_model, 2 * self.all_head_size, bias=config.attention_bias)
        else:
            self.Wqkv = nn.Linear(config.d_model, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
            rope_theta = config.local_rope_theta if config.local_rope_theta is not None else config.global_rope_theta
            self.max_position_embeddings = config.local_attention
        else:
            self.local_attention = (-1, -1)
            rope_theta = config.global_rope_theta

        if self.is_cross_attention:
            self.rotary_emb = None
        elif config._attn_implementation == "flash_attention_2":
            self.rotary_emb = VarWhisperFlashRotaryEmbedding(
                dim=self.head_dim, max_seqlen=self.max_position_embeddings, base=rope_theta
            )
        else:
            config_copy = copy.deepcopy(config)
            config_copy.rope_theta = rope_theta
            config_copy.hidden_size = config.d_model
            self.rotary_emb = VarWhisperRotaryEmbedding(config=config_copy, max_seqlen=self.max_position_embeddings)

        self.Wo = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        bs = hidden_states.shape[0]
        is_varlen = "cu_seqlens" in kwargs

        attn_func = partial(
            VARWHISPER_ATTENTION_FUNCTION[self.config._attn_implementation],
            module=self,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self.is_cross_attention:
            assert key_value_states is not None, "key_value_states must be provided for cross-attention"
            q = self.Wq(hidden_states)
            kv = self.Wkv(key_value_states)

            if is_varlen:
                q = q.view(-1, self.num_heads, self.head_dim)
                kv = kv.view(-1, 2, self.num_heads, self.head_dim)
            else:
                q = q.view(bs, -1, self.num_heads, self.head_dim)
                kv = kv.view(bs, -1, 2, self.num_heads, self.head_dim)

            attn_outputs = attn_func(qkv=q, kv=kv)
        else:
            assert key_value_states is None, "key_value_states must be None for self-attention"
            qkv = self.Wqkv(hidden_states)

            if is_varlen:
                qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
            else:
                qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

            attn_outputs = attn_func(qkv=qkv, rotary_emb=self.rotary_emb)

        hidden_states, *rest = attn_outputs
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return hidden_states, *rest


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class VarWhisperEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: VarWhisperConfig, layer_id: int):
        super().__init__()
        self.embed_dim: int = config.d_model

        self.self_attn = VarWhisperAttention(
            config=config,
            num_heads=config.encoder_attention_heads,
            max_position_embeddings=config.max_source_positions,
            layer_id=layer_id,
        )
        self.self_attn_layer_norm = nn.RMSNorm(self.embed_dim)
        self.dropout: float = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout: float = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.RMSNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class VarWhisperDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: VarWhisperConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim: int = config.d_model

        self.self_attn = VarWhisperAttention(
            config=config,
            num_heads=config.decoder_attention_heads,
            max_position_embeddings=config.max_target_positions,
            layer_id=layer_idx,
            is_causal=True,
        )
        self.dropout: float = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout: float = config.activation_dropout

        self.self_attn_layer_norm = nn.RMSNorm(self.embed_dim)
        self.cross_attn = VarWhisperAttention(
            config=config,
            num_heads=config.decoder_attention_heads,
            max_position_embeddings=config.max_target_positions,
            layer_id=layer_idx,
            is_cross_attention=True,
        )
        self.cross_attn_layer_norm = nn.RMSNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.RMSNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[EncoderDecoderCache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cu_seqlens: Optional[torch.LongTensor] = None,
            max_seqlen: Optional[int] = None,
            encoder_cu_seqlens: Optional[torch.LongTensor] = None,
            encoder_max_seqlen: Optional[int] = None,
            output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions
        )
        hidden_states = self_attn_outputs[0]
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_outputs = ()
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)
            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=past_key_value,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_k=encoder_cu_seqlens,
                max_seqlen_k=encoder_max_seqlen,
                output_attentions=output_attentions
            )
            hidden_states = cross_attn_outputs[0]
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + self_attn_outputs[1:] + cross_attn_outputs[1:]

        return outputs


class VarWhisperPreTrainedModel(PreTrainedModel):
    config_class = VarWhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VarWhisperEncoderLayer", "VarWhisperDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


class VarWhisperEncoder(VarWhisperPreTrainedModel):
    def __init__(self, config: VarWhisperConfig):
        super().__init__(config)
        self.dropout:float = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins: int = config.num_mel_bins
        self.padding_idx: int = config.pad_token_id
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList([VarWhisperEncoderLayer(config, layer_id=layer_id) for layer_id in range(config.encoder_layers)])
        self.layer_norm = nn.RMSNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
            self,
            input_features,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0).repeat(
            inputs_embeds.size(0), 1)

        hidden_states = inputs_embeds

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class VarWhisperDecoder(VarWhisperPreTrainedModel):
    main_input_name = "input_ids"

    def __init__(self, config: VarWhisperConfig):
        super().__init__(config)
        self.dropout: float = config.dropout
        self.padding_idx: int = config.pad_token_id
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.layers = nn.ModuleList(
            [VarWhisperDecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layer_norm = nn.RMSNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            past_key_values=None,
            inputs_embeds=None,
            position_ids=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            batch_size, seq_len = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        repad = False
        cu_seqlens, max_seqlen = None, None
        encoder_cu_seqlens, encoder_max_seqlen = None, None
        if self.config._attn_implementation == "flash_attention_2" and not use_cache:
            repad = True
            if inputs_embeds is None:
                with torch.no_grad():
                    input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_varwhisper_input(
                        inputs=input_ids, attention_mask=attention_mask
                    )
            else:
                inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_varwhisper_input(
                    inputs=inputs_embeds, attention_mask=attention_mask
                )

            if encoder_hidden_states is not None:
                encoder_hidden_states, encoder_cu_seqlens, encoder_max_seqlen = _unpad_encoder_hidden_states(
                    encoder_hidden_states.contiguous()
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        return_self_attention_cache = False
        if use_cache or past_key_values is not None:
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states: tuple = () if output_hidden_states else None
        all_self_attns: tuple = () if output_attentions else None
        all_cross_attentions: tuple = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_key_values if use_cache else None,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                encoder_cu_seqlens=encoder_cu_seqlens,
                encoder_max_seqlen=encoder_max_seqlen,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if repad:
            hidden_states = _pad_varwhisper_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_varwhisper_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    for hs in all_hidden_states
                )
        # If the attention implementation is FA2 and there is no need for repadding, there might still be the batch
        # dimension missing
        elif (
            self.config._attn_implementation == "flash_attention_2"
            and all_hidden_states is not None
            and all_hidden_states[-1].dim() == 2
        ):
            hidden_states = hidden_states.unsqueeze(0)
            all_hidden_states = tuple(hs.unsqueeze(0) for hs in all_hidden_states)

        next_cache = past_key_values if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            device: torch.device,
            cache_position: torch.Tensor,
            batch_size: int,
            **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class VarWhisperModel(VarWhisperPreTrainedModel):
    def __init__(self, config: VarWhisperConfig):
        super().__init__(config)

        self.encoder = VarWhisperEncoder(config)
        self.decoder = VarWhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]] = None,
            past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VarWhisperForConditionalGeneration(WhisperGenerationMixin, VarWhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: VarWhisperConfig):
        super().__init__(config)
        self.model = VarWhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            use_cache=None,
            encoder_outputs=None,
            attention_mask=None,
            decoder_attention_mask=None,
            cache_position=None,
            **kwargs,
    ):
        # Overwritten -- encoder-decoder whisper has custom logic, but it's close to the general function. Next time
        # this function needs to be touched, let's try to sort out the commonalities between the two and remove the
        # overwrite.

        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                decoder_position_ids = decoder_position_ids.clone(memory_format=torch.contiguous_format)

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1]:]

        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        decoder_input_ids = decoder_input_ids.contiguous()

        if (
                isinstance(past_key_values, EncoderDecoderCache)
                and (
                isinstance(past_key_values.self_attention_cache, StaticCache)
                or isinstance(past_key_values.cross_attention_cache, StaticCache)
        )
                and decoder_attention_mask is not None
                and decoder_attention_mask.ndim == 2
        ):
            batch_size, sequence_length = decoder_input_ids.shape

            decoder_attention_mask = self.get_decoder()._prepare_4d_causal_attention_mask_with_cache_position(
                decoder_attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.self_attention_cache.get_max_cache_shape(),
                dtype=self.proj_out.weight.dtype,
                device=decoder_input_ids.device,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "cache_position": cache_position,
        }