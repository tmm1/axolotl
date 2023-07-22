"""Flash attention monkey patch for llama model"""

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
# and https://github.com/jquesnelle/scaled-rope/blob/master/scaled_rope/modeling_llama.py
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from torch import nn

try:
    from flash_attn.flash_attn_interface import (  # pylint: disable=ungrouped-imports
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if self.use_flash_attention and not output_attentions:
        dtype = query_states.dtype
        # transform the data into the format required by flash attention
        qkv = torch.stack([query_states, key_states, value_states], dim=2).to(
            torch.float16
        )  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
        # We have disabled _prepare_decoder_attention_mask in LlamaModel
        # the attention_mask should be the same as the key_padding_mask
        key_padding_mask = attention_mask

        if key_padding_mask is None:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            max_s = q_len
            cu_q_lens = torch.arange(
                0,
                (bsz + 1) * q_len,
                step=q_len,
                dtype=torch.int32,
                device=qkv.device,
            )
            attn_output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            attn_output = rearrange(attn_output, "(b s) ... -> b s ...", b=bsz)
        else:
            nheads = qkv.shape[-2]

            # pylint: disable=invalid-name
            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(
                x_unpad,
                "nnz (three h d) -> nnz three h d",
                three=3,
                h=nheads,
            )
            output_unpad = flash_attn_varlen_qkvpacked_func(
                x_unpad,
                cu_q_lens,
                max_s,
                0.0,
                softmax_scale=None,
                causal=True,
            )
            attn_output = rearrange(
                pad_input(
                    rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                    indices,
                    bsz,
                    q_len,
                ),
                "b s (h d) -> b s h d",
                h=nheads,
            )
        attn_output = attn_output.transpose(1, 2).to(dtype)
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(  # pylint: disable=consider-using-generator
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
):  # pylint: disable=unused-argument
    # [bsz, seq_len]
    return attention_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (  # pylint: disable=protected-access
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
