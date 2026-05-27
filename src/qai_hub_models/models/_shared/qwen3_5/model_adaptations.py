# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
)

from qai_hub_models.models._shared.llm.common import TORCH_SUPPORTS_DYNAMIC_SHAPE
from qai_hub_models.models._shared.llm.model_adaptations import (
    ConvInplaceLinear,
    repeat_kv,
)
from qai_hub_models.models._shared.llm.sha_dynamic_kvcache import (
    SHADynamicCacheNewValueOnly,
)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d: nn.Conv1d,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(
        conv1d.weight.dtype
    )
    new_conv_state = hidden_states_new[:, :, -state_len:]
    out = conv1d(hidden_states_new)
    out = out[:, :, state_len : state_len + seq_len]
    if activation is not None:
        out = activation(out)
    out = out.to(hidden_states.dtype)
    return out, new_conv_state


def _apply_rope_single_partial(
    x: torch.Tensor, rope_vals: tuple[torch.Tensor, torch.Tensor], rotary_dim: int
) -> torch.Tensor:
    rope_real = rope_vals[0]
    rope_im = rope_vals[1]

    x_rot = x[:, :, :, :rotary_dim]
    x_pass = x[:, :, :, rotary_dim:]

    half_rot = rotary_dim // 2
    x_real = x_rot[:, :, :, :half_rot]
    x_im = x_rot[:, :, :, half_rot:]

    rope_real = rope_vals[0].to(dtype=x.dtype)
    rope_im = rope_vals[1].to(dtype=x.dtype)

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    x_rot_out = torch.cat((x_prod_real, x_prod_im), dim=3)
    return torch.cat((x_rot_out, x_pass), dim=3)


def QcQwen3_5_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: list[int] | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1] * 2
    query_states = _apply_rope_single_partial(q, (cos, sin), rotary_dim)
    key_states = _apply_rope_single_partial(k, (cos, sin), rotary_dim)
    return query_states, key_states


class SHAQwen3_5Attention(Qwen3_5Attention):
    """Split-Head Attention version of Qwen3.5 Attention (with Convs and gating).

    Key differences from Qwen3:
    - q_proj outputs 2x (query + gate), gate is applied via sigmoid after attention
    - Partial rotary embeddings (only partial_rotary_factor of head_dim)
    - q_norm and k_norm applied per-head
    - Uses (1+weight) style RMSNorm (Qwen3NextRMSNorm)
    """

    def prepare_conv(self) -> None:
        if not hasattr(self, "forward_no_conv"):
            self.q_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_attention_heads * self.head_dim * 2,
                1,
                bias=self.q_proj.bias is not None,
            )
            self.k_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                1,
                bias=self.k_proj.bias is not None,
            )
            self.v_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                1,
                bias=self.v_proj.bias is not None,
            )
            self.o_proj_conv = nn.Conv2d(
                self.config.num_attention_heads * self.head_dim,
                self.config.hidden_size,
                1,
                bias=self.o_proj.bias is not None,
            )

            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
            self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            if self.q_proj.bias is not None:
                assert self.q_proj_conv.bias is not None
                self.q_proj_conv.bias.data.copy_(self.q_proj.bias)
            if self.k_proj.bias is not None:
                assert self.k_proj_conv.bias is not None
                self.k_proj_conv.bias.data.copy_(self.k_proj.bias)
            if self.v_proj.bias is not None:
                assert self.v_proj_conv.bias is not None
                self.v_proj_conv.bias.data.copy_(self.v_proj.bias)
            if self.o_proj.bias is not None:
                assert self.o_proj_conv.bias is not None
                self.o_proj_conv.bias.data.copy_(self.o_proj.bias)

            del self.q_proj
            del self.k_proj
            del self.v_proj
            del self.o_proj

    def prepare_sha(self) -> None:
        if not (
            hasattr(self, "q_proj_conv")
            and hasattr(self, "k_proj_conv")
            and hasattr(self, "o_proj_conv")
            and hasattr(self, "v_proj_conv")
        ):
            raise RuntimeError(
                "The method 'prepare_sha' cannot be run on model without running 'prepare_conv' first."
            )

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        if not hasattr(self, "forward_mha"):
            self.q_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim * 2,
                        1,
                        bias=self.q_proj_conv.bias is not None,
                    )
                    for _ in range(num_heads)
                ]
            )
            self.k_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.k_proj_conv.bias is not None,
                    )
                    for _ in range(num_kv_heads)
                ]
            )
            self.v_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.v_proj_conv.bias is not None,
                    )
                    for _ in range(num_kv_heads)
                ]
            )

            self.q_norm_sha = nn.ModuleList(
                [
                    Qwen3_5RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(num_heads)
                ]
            )
            self.k_norm_sha = nn.ModuleList(
                [
                    Qwen3_5RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(num_kv_heads)
                ]
            )

            for i in range(num_heads):
                q_norm = self.q_norm_sha[i]
                assert isinstance(q_norm, Qwen3_5RMSNorm)
                q_norm.weight.data.copy_(self.q_norm.weight.data)
                assert hasattr(q_norm, "prepare_export")
                q_norm.prepare_export()
                assert getattr(q_norm, "_export_weight_folded", False)
            for i in range(num_kv_heads):
                k_norm = self.k_norm_sha[i]
                assert isinstance(k_norm, Qwen3_5RMSNorm)
                k_norm.weight.data.copy_(self.k_norm.weight.data)
                assert hasattr(k_norm, "prepare_export")
                k_norm.prepare_export()
                assert getattr(k_norm, "_export_weight_folded", False)

            self.forward_mha = cast(
                Callable[
                    [
                        torch.Tensor,
                        torch.Tensor | None,
                        torch.LongTensor | None,
                        Cache | None,
                        bool,
                        bool,
                        torch.LongTensor | None,
                        Any,
                    ],
                    tuple[
                        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None
                    ],
                ],
                self.forward,
            )
            self.forward = self.forward_sha  # type: ignore[assignment, unused-ignore]

        for i in range(num_heads):
            start_idx = i * self.head_dim * 2
            end_idx = (i + 1) * self.head_dim * 2
            q_proj = self.q_proj_sha[i]
            assert isinstance(q_proj, (nn.Linear, nn.Conv2d))
            q_proj.weight.data.copy_(self.q_proj_conv.weight[start_idx:end_idx, :])
            if self.q_proj_conv.bias is not None and q_proj.bias is not None:
                q_proj.bias.data.copy_(self.q_proj_conv.bias[start_idx:end_idx])

        for i in range(num_kv_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj = self.k_proj_sha[i]
            v_proj = self.v_proj_sha[i]
            assert isinstance(k_proj, (nn.Linear, nn.Conv2d))
            assert isinstance(v_proj, (nn.Linear, nn.Conv2d))
            k_proj.weight.data.copy_(self.k_proj_conv.weight[start_idx:end_idx, :])
            v_proj.weight.data.copy_(self.v_proj_conv.weight[start_idx:end_idx, :])
            if self.k_proj_conv.bias is not None and k_proj.bias is not None:
                k_proj.bias.data.copy_(self.k_proj_conv.bias[start_idx:end_idx])
            if self.v_proj_conv.bias is not None and v_proj.bias is not None:
                v_proj.bias.data.copy_(self.v_proj_conv.bias[start_idx:end_idx])

        del self.q_proj_conv
        del self.k_proj_conv
        del self.v_proj_conv
        del self.q_norm
        del self.k_norm

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        hidden_size = self.config.hidden_size
        num_kv_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        partial_rotary_factor = self.config.rope_parameters.get(
            "partial_rotary_factor", 1.0
        )
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            hidden_states = hidden_states.unsqueeze(2)
        else:
            hidden_states = torch.reshape(hidden_states, (bsz, -1, 1, hidden_size))
        hidden_states = hidden_states.transpose(1, 3)

        query_states = []
        gate_states = []
        for q_proj, q_norm in zip(self.q_proj_sha, self.q_norm_sha, strict=False):
            qg = q_proj(hidden_states).permute(0, 2, 3, 1)
            q, g = qg.chunk(2, dim=-1)
            q = q_norm(q)
            query_states.append(q)
            gate_states.append(g)

        key_states = [
            k_norm(k_proj(hidden_states).permute(0, 2, 3, 1))
            for k_proj, k_norm in zip(self.k_proj_sha, self.k_norm_sha, strict=False)
        ]
        value_states = [
            v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
        ]

        kv_seq_len = value_states[0].shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values.layers[self.layer_idx].values.shape[-2]

        assert position_embeddings is not None
        query_states = [
            _apply_rope_single_partial(q, position_embeddings, rotary_dim)
            for q in query_states
        ]
        key_states = [
            _apply_rope_single_partial(k, position_embeddings, rotary_dim)
            for k in key_states
        ]

        if past_key_values is not None:
            transposed_key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]

            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_values.update(
                torch.cat(key_states, dim=1),
                torch.cat(value_states, dim=1),
                self.layer_idx,
                cache_kwargs,
            )

            past_key = past_key_values.layers[self.layer_idx].keys
            past_value = past_key_values.layers[self.layer_idx].values
            key_states = [
                past_key[:, i, :, :].unsqueeze(1).transpose(2, 3).to(dtype=query_states[0].dtype)
                for i in range(past_key.shape[1])
            ]
            value_states = [
                past_value[:, i, :, :].unsqueeze(1).to(dtype=query_states[0].dtype)
                for i in range(past_value.shape[1])
            ]
        else:
            key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]

        key_states = list(repeat_kv(key_states, num_kv_groups))
        value_states = list(repeat_kv(value_states, num_kv_groups))

        attn_weights = [
            torch.matmul(q, k / math.sqrt(self.head_dim))
            for q, k in zip(query_states, key_states, strict=False)
        ]

        if attention_mask is not None:
            attn_weights = [aw + attention_mask for aw in attn_weights]

        attn_weights = [
            nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
                query_states[0].dtype
            )
            for aw in attn_weights
        ]
        attn_weights = [
            nn.functional.dropout(aw, p=self.attention_dropout, training=self.training)
            for aw in attn_weights
        ]
        attn_output = [
            torch.matmul(aw, v)
            for aw, v in zip(attn_weights, value_states, strict=False)
        ]

        attn_output = [
            ao * torch.sigmoid(g)
            for ao, g in zip(attn_output, gate_states, strict=False)
        ]

        attn_output_return: torch.Tensor = torch.cat(attn_output, dim=3)
        attn_output_return = attn_output_return.permute(0, 3, 1, 2)
        attn_output_return = self.o_proj_conv(attn_output_return)
        attn_output_return = attn_output_return.transpose(1, 3)
        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            attn_output_return = attn_output_return.squeeze(2)
        else:
            attn_output_return = attn_output_return.reshape(bsz, q_len, hidden_size)

        attn_weights_return = attn_weights if output_attentions else None

        assert Version(transformers.__version__) >= Version("4.48.0")
        return attn_output_return, attn_weights_return


class QCQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    Adapted GatedDeltaNet for explicit state I/O (needed for ONNX export).

    Instead of relying on DynamicCache for in-place state updates,
    this module accepts conv_state and recurrent_state as explicit inputs
    and returns updated states as outputs.
    """

    def prepare_export(self) -> None:
        if getattr(self, "_export_a_log_folded", False):
            return
        self.A_log.data.copy_((-self.A_log.float().exp()).to(dtype=self.A_log.dtype))
        self._export_a_log_folded = True

    def prepare_conv(self) -> None:
        if self.conv1d.bias is None:
            conv1d_with_bias = nn.Conv1d(
                in_channels=self.conv1d.in_channels,
                out_channels=self.conv1d.out_channels,
                kernel_size=self.conv1d.kernel_size,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv1d.groups,
                bias=True,
                padding_mode=self.conv1d.padding_mode,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            conv1d_with_bias.weight.data.copy_(self.conv1d.weight.data)
            assert conv1d_with_bias.bias is not None
            conv1d_with_bias.bias.data.zero_()
            self.conv1d = conv1d_with_bias
        if not isinstance(self.in_proj_qkv, ConvInplaceLinear):
            self.in_proj_qkv = ConvInplaceLinear(self.in_proj_qkv)
        if not isinstance(self.in_proj_z, ConvInplaceLinear):
            self.in_proj_z = ConvInplaceLinear(self.in_proj_z)
        if not isinstance(self.in_proj_b, ConvInplaceLinear):
            self.in_proj_b = ConvInplaceLinear(self.in_proj_b)
        if not isinstance(self.in_proj_a, ConvInplaceLinear):
            self.in_proj_a = ConvInplaceLinear(self.in_proj_a)
        if not isinstance(self.out_proj, ConvInplaceLinear):
            self.out_proj = ConvInplaceLinear(self.out_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        conv_state: torch.Tensor | None = None
        recurrent_state: torch.Tensor | None = None
        if cache_params is not None:
            if hasattr(cache_params, "conv_states") and len(cache_params.conv_states) > self.layer_idx:
                conv_state = cache_params.conv_states[self.layer_idx]
            if hasattr(cache_params, "recurrent_states") and len(cache_params.recurrent_states) > self.layer_idx:
                recurrent_state = cache_params.recurrent_states[self.layer_idx]

            if conv_state is None and hasattr(cache_params, "layers") and len(cache_params.layers) > self.layer_idx:
                layer_cache = cache_params.layers[self.layer_idx]
                if hasattr(layer_cache, "conv_states"):
                    conv_state = layer_cache.conv_states
                if hasattr(layer_cache, "recurrent_states"):
                    recurrent_state = layer_cache.recurrent_states

        if conv_state is None:
            conv_dim = self.key_dim * 2 + self.value_dim
            conv_state = torch.zeros(
                batch_size,
                conv_dim,
                self.conv_kernel_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        output, new_conv_state, new_recurrent_state = self.forward_explicit_state(
            hidden_states=hidden_states,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            attention_mask=attention_mask,
        )

        if cache_params is not None:
            if hasattr(cache_params, "update_conv_state"):
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if hasattr(cache_params, "update_recurrent_state"):
                cache_params.update_recurrent_state(
                    new_recurrent_state, self.layer_idx
                )

        return output

    def forward_explicit_state(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_state = conv_state.to(dtype=hidden_states.dtype)
        recurrent_state = recurrent_state.to(dtype=hidden_states.dtype)

        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        # Use fused projections (no SHA splitting for GatedDeltaNet)
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        mixed_qkv, new_conv_state = torch_causal_conv1d_update(
            hidden_states=mixed_qkv,
            conv_state=conv_state,
            conv1d=self.conv1d,
            activation=F.silu,
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query = mixed_qkv[..., : self.key_dim]
        key = mixed_qkv[..., self.key_dim : self.key_dim * 2]
        value = mixed_qkv[..., self.key_dim * 2 : self.key_dim * 2 + self.value_dim]

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = self.A_log * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, new_recurrent_state = _torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output, new_conv_state, new_recurrent_state


class QCQwen3_5MLP(Qwen3_5MLP):
    def prepare_conv(self) -> None:
        if not isinstance(self.gate_proj, ConvInplaceLinear):
            self.gate_proj = ConvInplaceLinear(self.gate_proj)
        if not isinstance(self.up_proj, ConvInplaceLinear):
            self.up_proj = ConvInplaceLinear(self.up_proj)
        if not isinstance(self.down_proj, ConvInplaceLinear):
            self.down_proj = ConvInplaceLinear(self.down_proj)


class QCQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def prepare_conv(self) -> None:
        self.lm_head = ConvInplaceLinear(self.lm_head)


def patched_qwen3_5_decoder_layer_forward(
    self: Qwen3_5DecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    **kwargs: Any,
) -> torch.FloatTensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        linear_attn = cast(QCQwen3_5GatedDeltaNet, self.linear_attn)
        hidden_states = linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported layer type: {self.layer_type}")

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def patched_qwen3_5_text_model_forward(
    self: Qwen3_5TextModel,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: Any = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Any,
) -> Any:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ModelOutputWithPast,
        create_causal_mask,
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = SHADynamicCacheNewValueOnly(config=self.config)

    linear_attn_source_mask = _normalize_linear_attention_mask(
        attention_mask, past_key_values
    )

    if isinstance(position_ids, (tuple, list)) and len(position_ids) == 2:
        position_embeddings = tuple(position_ids)
        position_ids = None
        text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = linear_attn_source_mask
    else:
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                )
                + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                4, position_ids.shape[0], -1
            )

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = linear_attn_source_mask

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds

    for i, decoder_layer in enumerate(
        self.layers[: self.config.num_hidden_layers]
    ):
        layer_mask = (
            linear_attn_mask
            if self.config.layer_types[i] == "linear_attention"
            else causal_mask
        )
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _normalize_linear_attention_mask(
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
) -> torch.Tensor | None:
    del past_key_values

    if attention_mask is None:
        return None

    linear_attn_mask = attention_mask
    if attention_mask.ndim == 4:
        query_length = attention_mask.shape[2]
        linear_attn_mask = (
            attention_mask.squeeze(1).amax(dim=1) >= 0
        ).to(attention_mask.dtype)
        linear_attn_mask = linear_attn_mask[:, -query_length:]

    return linear_attn_mask


def _apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 2:
        hidden_states = hidden_states * attention_mask[:, :, None].to(
            hidden_states.dtype
        )
    return hidden_states


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out_list = []
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

        core_attn_out_list.append(
            (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        )

    core_attn_out = torch.stack(core_attn_out_list, dim=2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = (
        (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()
    ).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
