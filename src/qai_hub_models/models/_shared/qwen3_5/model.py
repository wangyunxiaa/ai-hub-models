# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    PositionProcessorBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import json
import logging
import os
from collections.abc import Collection
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen3_5 import modeling_qwen3_5

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.sha_dynamic_kvcache import (
    SHADynamicCacheNewValueOnly,
)
from qai_hub_models.models._shared.qwen3_5.model_adaptations import (
    QcQwen3_5_apply_rotary_pos_emb,
    QCQwen3_5ForCausalLM,
    QCQwen3_5GatedDeltaNet,
    QCQwen3_5MLP,
    SHAQwen3_5Attention,
    patched_qwen3_5_decoder_layer_forward,
    patched_qwen3_5_text_model_forward,
)
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.input_spec import InputSpec

import re as _re


class StateSchemaMode(str, Enum):
    KV = "kv"
    HYBRID = "hybrid"


_STATE_NAME_PATTERNS = {
    "past_key": _re.compile(r"^past_key_(\d+)_(in|out)$"),
    "past_value": _re.compile(r"^past_value_(\d+)_(in|out)$"),
    "conv_state": _re.compile(r"^conv_state_(\d+)_(in|out)$"),
    "recurrent_state": _re.compile(r"^recurrent_state_(\d+)_(in|out)$"),
}
_LINEAR_STATE_KINDS = {"conv_state", "recurrent_state"}


def _is_state_tensor_name(
    name: str,
    state_schema_mode: StateSchemaMode = StateSchemaMode.HYBRID,
) -> bool:
    for state_kind, pattern in _STATE_NAME_PATTERNS.items():
        match = pattern.match(name)
        if match is None:
            continue
        if state_schema_mode == StateSchemaMode.KV and state_kind in _LINEAR_STATE_KINDS:
            return False
        return True
    return False


def get_state_tensor_names(
    names: list[str],
    state_schema_mode: StateSchemaMode = StateSchemaMode.HYBRID,
) -> list[str]:
    return [name for name in names if _is_state_tensor_name(name, state_schema_mode)]


def shift_kv_state_tensors(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    ret = []
    if len(past_key_vals) == 0:
        for i in range(0, len(new_key_vals), 2):
            orig_key_shape = new_key_vals[i].shape
            key_shape = (orig_key_shape[0], orig_key_shape[1], orig_key_shape[2], 0)
            past_key_vals.append(torch.zeros(key_shape, device=device))
            orig_value_shape = new_key_vals[i + 1].shape
            value_shape = (orig_value_shape[0], orig_value_shape[1], 0, orig_value_shape[3])
            past_key_vals.append(torch.zeros(value_shape, device=device))
    if len(new_key_vals) == 0:
        for i in range(0, len(past_key_vals), 2):
            orig_key_shape = past_key_vals[i].shape
            key_shape = (orig_key_shape[0], orig_key_shape[1], orig_key_shape[2], 0)
            new_key_vals.append(torch.zeros(key_shape, device=device))
            orig_value_shape = past_key_vals[i + 1].shape
            value_shape = (orig_value_shape[0], orig_value_shape[1], 0, orig_value_shape[3])
            new_key_vals.append(torch.zeros(value_shape, device=device))
    for i in range(0, len(past_key_vals), 2):
        key_cache = torch.cat([past_key_vals[i].to(device), new_key_vals[i].to(device)], dim=3)
        key_cache = key_cache[:, :, :, -length:]
        val_cache = torch.cat([past_key_vals[i + 1].to(device), new_key_vals[i + 1].to(device)], dim=2)
        val_cache = val_cache[:, :, -length:, :]
        ret.append(key_cache)
        ret.append(val_cache)
    return ret

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_qwen35"

DATA_DIR = "data"
USE_CACHED_DATA = True

# Qwen3.5 uses the same ChatML format as Qwen3
START_HEADER = "<|im_start|>"
END_HEADER = "<|im_end|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
END_TOKENS = {"<|im_end|>", "<|endoftext|>"}


class Qwen3_5_Optimizations(str, Enum):
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class Qwen3_5RopeEmbedding:
    """
    Position embedding for Qwen3.5 with partial rotary factor.

    Only partial_rotary_factor of head_dim is used for RoPE,
    so the compact cos/sin have shape (1, 1, seq_len, rotary_dim//2).
    """

    def __init__(
        self,
        max_length: int = 4096,
        config: PretrainedConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("config is required for Qwen3_5RopeEmbedding")

        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        rope_params = getattr(config, "rope_parameters", None) or {}
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        rope_theta = rope_params.get("rope_theta", 10000.0)

        rotary_dim = int(head_dim * partial_rotary_factor)

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float)
                / rotary_dim
            )
        )

        # Precompute cos/sin for all positions up to max_length
        positions = torch.arange(max_length, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)  # (max_length, rotary_dim/2)

        # Store in compact format: (1, 1, max_length, rotary_dim//2)
        self.cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        self.sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (batch_size, sequence_length)
        Returns: (cos, sin) each of shape (batch_size, 1, sequence_length, rotary_dim//2)
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)
        sin = self.sin[0, 0, :, :].to(position_ids.device)
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class Qwen3_5Base(LLMBase):
    state_schema_mode = StateSchemaMode.HYBRID
    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds
    LMClass = modeling_qwen3_5.Qwen3_5ForCausalLM
    EmbeddingClass = Qwen3_5RopeEmbedding

    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        # Force float32 to avoid dtype mismatch in GatedDeltaNet conv1d.
        # The model config defaults to bfloat16, which causes issues with
        # conv1d ops in linear attention layers during FP evaluation.
        llm_config.torch_dtype = torch.float32
        if hasattr(llm_config, "text_config"):
            llm_config.text_config.torch_dtype = torch.float32
        return llm_config

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        return {
            "global_prefix": "",
            "system_prefix": f"{START_HEADER}{SYSTEM_ID}\n",
            "system_suffix": f"{END_HEADER}\n",
            "user_prefix": f"{START_HEADER}{USER_ID}\n",
            "user_suffix": f"{END_HEADER}\n",
            "assistant_prefix": f"{START_HEADER}{ASSISTANT_ID}\n",
            "assistant_suffix": f"{END_HEADER}\n",
            "default_system_prompt": cls.default_system_prompt,
        }

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Qwen3_5_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            modeling_qwen3_5.Qwen3_5Attention = SHAQwen3_5Attention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_qwen3_5.Qwen3_5TextRotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(
            modeling_qwen3_5.Qwen3_5TextRotaryEmbedding, "_original_forward"
        ):
            modeling_qwen3_5.Qwen3_5TextRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.forward
            )
            modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.forward = (
                bypass_RotaryEmbedding
            )
        modeling_qwen3_5.apply_rotary_pos_emb = QcQwen3_5_apply_rotary_pos_emb  # type: ignore[attr-defined, unused-ignore]

        def _prepare_qwen3_5_rmsnorm_export(self: modeling_qwen3_5.Qwen3_5RMSNorm) -> None:
            if getattr(self, "_export_weight_folded", False):
                return
            self.weight.data.add_(1.0)
            self._export_weight_folded = True

        def Qwen3_5RMSNorm_forward(
            self: modeling_qwen3_5.Qwen3_5RMSNorm, hidden_states: torch.Tensor
        ) -> torch.Tensor:
            added_dims = max(0, 4 - hidden_states.dim())
            for _ in range(added_dims):
                hidden_states = hidden_states.unsqueeze(0)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            eps = getattr(self, "variance_epsilon", self.eps)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            assert getattr(self, "_export_weight_folded", False), (
                "Qwen3.5 RMSNorm weight must be folded before forward/export."
            )
            hidden_states = hidden_states * self.weight
            for _ in range(added_dims):
                hidden_states = hidden_states.squeeze(0)
            return hidden_states

        if (
            skip_optimizations
            and Qwen3_5_Optimizations.RMS_NORM_4_RANK in skip_optimizations
        ):
            print("Skip rank4_rms_norm optimization")
        else:
            modeling_qwen3_5.Qwen3_5RMSNorm.forward = Qwen3_5RMSNorm_forward
            modeling_qwen3_5.Qwen3_5RMSNorm.prepare_export = _prepare_qwen3_5_rmsnorm_export  # type: ignore[attr-defined, unused-ignore]

        modeling_qwen3_5.Qwen3_5MLP = QCQwen3_5MLP  # type: ignore[misc, unused-ignore]
        modeling_qwen3_5.Qwen3_5ForCausalLM = QCQwen3_5ForCausalLM  # type: ignore[misc, unused-ignore]
        modeling_qwen3_5.Qwen3_5GatedDeltaNet = QCQwen3_5GatedDeltaNet  # type: ignore[misc, unused-ignore]
        Qwen3_5Base.LMClass = QCQwen3_5ForCausalLM

        modeling_qwen3_5.Qwen3_5DecoderLayer.forward = patched_qwen3_5_decoder_layer_forward  # type: ignore[assignment, unused-ignore]
        modeling_qwen3_5.Qwen3_5TextModel.forward = patched_qwen3_5_text_model_forward  # type: ignore[assignment, unused-ignore]

    def _verify_ckpt(self) -> None:
        architectures = getattr(self.llm_config, "architectures", None) or []
        arch_ok = len(architectures) == 0 or any(
            arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
            for arch in architectures
        )
        if not (
            arch_ok
            and self.llm_config.model_type in ("qwen3_5_text", "qwen3_5")
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )

    def _get_layer_types(self) -> list[str]:
        """Get the layer types from config."""
        if hasattr(self.llm_config, "text_config"):
            config = self.llm_config.text_config
        else:
            config = self.llm_config
        return getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers

    def _get_text_config(self) -> PretrainedConfig:
        """Get the text sub-config (handles both standalone and VL configs)."""
        if hasattr(self.llm_config, "text_config"):
            return self.llm_config.text_config
        return self.llm_config

    def _get_linear_attn_config(self) -> dict[str, int]:
        """Get linear attention configuration parameters."""
        text_config = self._get_text_config()
        return {
            "linear_conv_kernel_dim": getattr(text_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(text_config, "linear_key_head_dim", 128),
            "linear_value_head_dim": getattr(text_config, "linear_value_head_dim", 128),
            "linear_num_key_heads": getattr(text_config, "linear_num_key_heads", 16),
            "linear_num_value_heads": getattr(text_config, "linear_num_value_heads", 16),
        }

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            position_ids = rest[0]
            state_tensors = rest[1:]
        else:
            position_ids = rest[:2]
            state_tensors = rest[2:]

        layer_types = self._get_layer_types()
        text_config = self._get_text_config()
        expected_num_state_tensors = len(layer_types) * 2
        if len(state_tensors) != expected_num_state_tensors:
            raise ValueError(
                f"Expected {expected_num_state_tensors} hybrid state tensors, got {len(state_tensors)}."
            )

        cache = SHADynamicCacheNewValueOnly(config=text_config)

        tensor_idx = 0
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                past_key = state_tensors[tensor_idx]
                past_value = state_tensors[tensor_idx + 1]
                tensor_idx += 2

                k = past_key.permute(1, 0, 3, 2)
                v = past_value.permute(1, 0, 2, 3)
                cache.update(k, v, layer_idx)
            else:
                conv_state = state_tensors[tensor_idx]
                recurrent_state = state_tensors[tensor_idx + 1]
                tensor_idx += 2
                cache.update_conv_state(conv_state, layer_idx)
                cache.update_recurrent_state(recurrent_state, layer_idx)

        model_kwargs: dict[str, Any] = {
            self.main_input_name: input_tokens,
            "attention_mask": self.attention_mask_multiplier * attention_mask,
            "position_ids": position_ids,
            "past_key_values": cache,
        }
        out = self.model(**model_kwargs)

        out_cache = out["past_key_values"]
        flat_output_states: list[torch.Tensor] = []

        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                if hasattr(out_cache, "key_cache"):
                    keys = out_cache.key_cache[layer_idx]
                    values = out_cache.value_cache[layer_idx]
                else:
                    keys = out_cache.layers[layer_idx].keys
                    values = out_cache.layers[layer_idx].values

                k_out = keys[:, :, -self.sequence_length:, :].permute(1, 0, 3, 2)
                v_out = values[:, :, -self.sequence_length:, :].permute(1, 0, 2, 3)
                flat_output_states.append(k_out)
                flat_output_states.append(v_out)
            else:
                if hasattr(out_cache, "conv_states"):
                    conv_state_out = out_cache.conv_states[layer_idx]
                    recurrent_state_out = out_cache.recurrent_states[layer_idx]
                else:
                    layer_cache = out_cache.layers[layer_idx]
                    conv_state_out = layer_cache.conv_states
                    recurrent_state_out = layer_cache.recurrent_states
                flat_output_states.append(conv_state_out)
                flat_output_states.append(recurrent_state_out)

        return [out["logits"], *flat_output_states]

    @staticmethod
    def _get_output_names(
        num_hidden_layers: int,
        layer_types: list[str] | None = None,
        kv_only: bool = False,
    ) -> list[str]:
        """
        Generate output names for the hybrid model.

        For full_attention layers: past_key_{i}_out, past_value_{i}_out
        For linear_attention layers: conv_state_{i}_out, recurrent_state_{i}_out

        If kv_only=True, only include outputs for full_attention layers (for FP eval).
        """
        output_names = ["logits"]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                output_names.append(f"past_key_{i}_out")
                output_names.append(f"past_value_{i}_out")
            elif not kv_only:
                output_names.append(f"conv_state_{i}_out")
                output_names.append(f"recurrent_state_{i}_out")
        return output_names

    @staticmethod
    def _get_input_spec_hybrid(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int,
        layer_types: list[str],
        linear_attn_config: dict[str, int],
        partial_rotary_factor: float = 0.25,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
        kv_only: bool = False,
    ) -> InputSpec:
        """
        Build input spec for hybrid model with both full_attention and linear_attention layers.

        If kv_only=True, only include KV cache entries for full_attention layers (for FP eval).
        Linear attention state is managed internally by the model in that case.
        """
        rotary_dim = int(head_dim * partial_rotary_factor)
        embed_dim = rotary_dim // 2
        input_spec: InputSpec = {}

        # Primary input
        if llm_io_type == LLMIOType.genie_input_embeds:
            input_spec["inputs_embeds"] = ((1, sequence_length, hidden_size), "float32")
        else:
            input_spec["input_ids"] = ((1, sequence_length), "int32")

        # Attention mask
        input_spec["attention_mask"] = (
            (1, 1, sequence_length, context_length),
            "float32",
        )

        # Position IDs
        if llm_io_type == LLMIOType.huggingface_input_ids:
            input_spec["position_ids"] = ((1, sequence_length), "int32")
        else:
            input_spec["position_ids_cos"] = (
                (1, 1, sequence_length, embed_dim),
                "float32",
            )
            input_spec["position_ids_sin"] = (
                (1, 1, sequence_length, embed_dim),
                "float32",
            )

        # Per-layer state inputs
        conv_kernel_dim = linear_attn_config["linear_conv_kernel_dim"]
        key_dim = (
            linear_attn_config["linear_num_key_heads"]
            * linear_attn_config["linear_key_head_dim"]
        )
        value_dim = (
            linear_attn_config["linear_num_value_heads"]
            * linear_attn_config["linear_value_head_dim"]
        )
        conv_dim = key_dim * 2 + value_dim
        num_v_heads = linear_attn_config["linear_num_value_heads"]
        k_head_dim = linear_attn_config["linear_key_head_dim"]
        v_head_dim = linear_attn_config["linear_value_head_dim"]

        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                # Standard KV cache
                input_spec[f"past_key_{i}_in"] = (
                    (num_key_value_heads, 1, head_dim, context_length - sequence_length),
                    "float32",
                )
                input_spec[f"past_value_{i}_in"] = (
                    (num_key_value_heads, 1, context_length - sequence_length, head_dim),
                    "float32",
                )
            elif not kv_only:
                input_spec[f"conv_state_{i}_in"] = (
                    (1, conv_dim, conv_kernel_dim),
                    "float32",
                )
                input_spec[f"recurrent_state_{i}_in"] = (
                    (1, num_v_heads, k_head_dim, v_head_dim),
                    "float32",
                )

        return input_spec


class Qwen3_5PositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: PretrainedConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = Qwen3_5RopeEmbedding(max_length=self.context_len, config=config)

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask_converter = AttentionMaskConverter(True)
        attention_mask = attention_mask_converter.to_4d(
            attention_mask_before_processor,
            query_length=position_ids.shape[1],
            key_value_length=attention_mask_before_processor.shape[1],
            dtype=torch.float32,
        )
        attention_mask = attention_mask.clip(-1000, 0)
        return attention_mask, position_ids_cos, position_ids_sin


class Qwen3_5Base_AIMETOnnx(LLM_AIMETOnnx):
    state_schema_mode = StateSchemaMode.HYBRID
    EmbeddingClass = Qwen3_5RopeEmbedding
    FPModel = Qwen3_5Base
    split_embedding = False

    ada_scale_model_type: str | None = "qwen3"

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        return (-1000.0, 1.0)

    def __init__(
        self,
        quant_sim: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            quant_sim=quant_sim,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )
        self._linear_attn_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_layer_types(self) -> list[str]:
        if hasattr(self.llm_config, "text_config"):
            config = self.llm_config.text_config
        else:
            config = self.llm_config
        return getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers

    def _get_text_config(self) -> PretrainedConfig:
        if hasattr(self.llm_config, "text_config"):
            return self.llm_config.text_config
        return self.llm_config

    def _get_linear_attn_config(self) -> dict[str, int]:
        text_config = self._get_text_config()
        return {
            "linear_conv_kernel_dim": getattr(text_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(text_config, "linear_key_head_dim", 128),
            "linear_value_head_dim": getattr(text_config, "linear_value_head_dim", 128),
            "linear_num_key_heads": getattr(text_config, "linear_num_key_heads", 16),
            "linear_num_value_heads": getattr(text_config, "linear_num_value_heads", 16),
        }

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> "DatasetEntries | None":
        import math

        import numpy as np
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from qai_hub_models.datasets import get_dataset_from_name
        from qai_hub_models.datasets.common import DatasetSplit
        from qai_hub_models.models._shared.llm.generator import LLM_Generator
        from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries
        from qai_hub_models.utils.runtime_torch_wrapper import kwargs_to_dict

        if num_samples == 0:
            num_samples = math.ceil(80000 / self.context_length)

        dataset = get_dataset_from_name(
            name="wikitext",
            split=DatasetSplit.TRAIN,
            tokenizer=self.tokenizer,
            block_size=self.sequence_length,
            context_length=self.context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None

        assert self.EmbeddingClass is not None
        rope_embeddings = self.EmbeddingClass(
            max_length=self.context_length, config=self.llm_config
        )
        generator = LLM_Generator(
            [self],
            self.tokenizer,
            rope_embeddings,
        )

        all_input_names = list(input_spec.keys())
        mamba_state_names = [
            k for k in all_input_names
            if k.startswith("conv_state_") or k.startswith("recurrent_state_")
        ]
        non_mamba_names = [
            k for k in all_input_names
            if k not in mamba_state_names
        ]

        with self.remove_quantization():
            all_inputs: dict[str, list[np.ndarray]] | None = None
            for sample in tqdm(
                dataloader, total=len(dataloader), desc="Pre-filling calibration data"
            ):
                input_ids, attention_mask, _ = sample
                self._linear_attn_cache.clear()
                for prefilled_inputs in generator.prefill(input_ids, attention_mask):
                    if all_inputs is None:
                        all_inputs = {name: [] for name in all_input_names}

                    non_mamba_dict = kwargs_to_dict(non_mamba_names, *prefilled_inputs)
                    for name in non_mamba_names:
                        all_inputs[name].append(
                            non_mamba_dict[name].cpu().detach().numpy()
                            if isinstance(non_mamba_dict[name], torch.Tensor)
                            else non_mamba_dict[name]
                        )

                    for name in mamba_state_names:
                        layer_idx = int(name.split("_")[2])
                        if layer_idx in self._linear_attn_cache:
                            conv_state, recurrent_state = self._linear_attn_cache[layer_idx]
                            if "conv_state" in name:
                                all_inputs[name].append(conv_state.numpy())
                            else:
                                all_inputs[name].append(recurrent_state.numpy())
                        else:
                            shape, dtype = input_spec[name]
                            all_inputs[name].append(np.zeros(shape, dtype=dtype))

        assert all_inputs is not None
        tensors_tuple = tuple(all_inputs[name] for name in all_input_names)
        return make_hub_dataset_entries(tensors_tuple, all_input_names)

    @staticmethod
    def _get_output_names(
        num_hidden_layers: int,
        layer_types: list[str] | None = None,
    ) -> list[str]:
        output_names = ["logits"]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                output_names.append(f"past_key_{i}_out")
                output_names.append(f"past_value_{i}_out")
            else:
                output_names.append(f"conv_state_{i}_out")
                output_names.append(f"recurrent_state_{i}_out")
        return output_names

    @classmethod
    def create_quantsim(
        cls,
        onnx_model: onnx.ModelProto,
        host_device: torch.device,
        precision: Precision,
    ) -> "QuantizationSimModel":
        quant_sim = cls._build_quantsim(
            onnx_model, cls.get_ort_providers(host_device), precision
        )
        return cls._configure_quant_sim(quant_sim, precision)

    @staticmethod
    def _build_quantsim(
        onnx_model: onnx.ModelProto,
        providers: list[str | tuple[str, dict]],
        precision: Precision = Precision.w4a16,
    ) -> "QuantizationSimModel":
        from aimet_onnx import quantsim
        from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme
        from aimet_onnx.common import quantsim as qs
        from aimet_onnx.quantsim import AimetLogger

        AimetLogger.set_level_for_all_areas(logging.WARNING)
        default_config = get_aimet_config_path("default_config_qwen35")
        quantsim.op_types_to_tie_qtzrs = ["Concat"]
        quantsim._tie_qtzrs = True
        quantsim.op_outputs_to_ignore.append("Slice")
        quantsim.op_outputs_to_ignore.append("Constant")
        qs.encoding_version = "1.0.0"

        if precision == Precision.w8a16:
            param_type = "int8"
        else:
            param_type = "int4"

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            param_type=param_type,
            activation_type="int16",
            quant_scheme=QuantScheme.min_max,
            config_file=default_config,
            providers=providers,
        )
        print(f"QuantSim session providers: {quant_sim.session.get_providers()}")
        return quant_sim

    @classmethod
    def _configure_quant_sim(
        cls, quant_sim: "QuantizationSimModel", precision: Precision
    ) -> "QuantizationSimModel":
        from aimet_onnx.common.defs import QuantizationDataType
        from qai_hub_models.models._shared.llm._utils import (
            _apply_int8_kv_cache_tying_and_lm_head,
            _get_kv_io_map,
            _set_lm_head_to_8b,
        )

        if precision == Precision.w8a16:
            kv_io_map = _get_kv_io_map(quant_sim)
            quant_sim = _apply_int8_kv_cache_tying_and_lm_head(quant_sim, kv_io_map)
            cls._set_mamba_states_to_float16(quant_sim)
        elif precision == Precision.w4a16:
            kv_io_map = _get_kv_io_map(quant_sim)
            quant_sim = _apply_int8_kv_cache_tying_and_lm_head(quant_sim, kv_io_map)
            cls._set_mamba_states_to_float16(quant_sim)
            cls._set_int4_weights_to_per_block(quant_sim, block_size=32)
        elif precision == Precision.w4:
            _set_lm_head_to_8b(quant_sim)
            cls._set_mamba_states_to_float16(quant_sim)
            cls._set_int4_weights_to_per_block(quant_sim, block_size=32)
            for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
                if op_name in quant_sim.activation_names:
                    qc_op.reset_encoding_stats()
                    qc_op.data_type = QuantizationDataType.float
                    qc_op.bitwidth = 16
        return quant_sim

    @staticmethod
    def _set_mamba_states_to_float16(quant_sim: "QuantizationSimModel") -> None:
        from aimet_onnx.common.defs import QuantizationDataType

        mamba_state_names = set()
        for inp in quant_sim.model.graph().input:
            if "conv_state" in inp.name or "recurrent_state" in inp.name:
                mamba_state_names.add(inp.name)
        for out in quant_sim.model.graph().output:
            name = out.name.replace("_updated", "")
            if "conv_state" in name or "recurrent_state" in name:
                mamba_state_names.add(name)

        for name in mamba_state_names:
            quantizer = quant_sim.qc_quantize_op_dict.get(name)
            if quantizer is not None and quantizer.enabled:
                quantizer.reset_encoding_stats()
                quantizer.data_type = QuantizationDataType.float
                quantizer.bitwidth = 16

    @staticmethod
    def _set_int4_weights_to_per_block(
        quant_sim: "QuantizationSimModel", block_size: int = 32
    ) -> None:
        from aimet_onnx.quantsim import set_blockwise_quantization_for_weights

        set_blockwise_quantization_for_weights(
            sim=quant_sim,
            op_types=("Conv", "Gemm", "MatMul"),
            bitwidth=4,
            symmetric=True,
            block_size=block_size,
        )

    def _dataloader_to_numpy(
        self, data, num_batches: int
    ) -> list[dict[str, Any]]:
        import numpy as np
        from tqdm import tqdm
        import itertools
        from qai_hub_models.utils.runtime_torch_wrapper import kwargs_to_dict

        assert self.quant_sim is not None
        session = self.quant_sim.session
        onnx_input_names = [inp.name for inp in session.get_inputs()]

        calib_input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            llm_io_type=self.llm_io_type,
        )
        calib_input_names = list(calib_input_spec.keys())

        onnx_data = []
        n = min(len(data), num_batches)
        for batch in tqdm(itertools.islice(data, n), total=n):
            batch_list = list(batch)
            calib_dict = kwargs_to_dict(calib_input_names, *batch_list)
            entry: dict[str, Any] = {}
            for name in onnx_input_names:
                if name in calib_dict:
                    val = calib_dict[name]
                    if isinstance(val, torch.Tensor):
                        entry[name] = val.cpu().detach().numpy()
                    elif isinstance(val, np.ndarray):
                        entry[name] = val
                    else:
                        entry[name] = np.array(val)
                else:
                    inp_info = next(inp for inp in session.get_inputs() if inp.name == name)
                    shape = [d if isinstance(d, int) else 1 for d in inp_info.shape]
                    entry[name] = np.zeros(shape, dtype=np.float32)
            onnx_data.append(entry)
        return onnx_data

    @classmethod
    def prepare_genie_assets(
        cls,
        hub_device: hub.Device,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        context_lengths: list[int],
        model_list: list[str],
        output_path: Path,
        precision: Precision,
        encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
        model_id: str,
        model_name: str,
    ) -> None:
        super().prepare_genie_assets(
            hub_device,
            checkpoint,
            llm_config,
            context_lengths,
            model_list,
            output_path,
            precision,
            encodings_path,
            input_specs,
            output_specs,
            model_id=model_id,
            model_name=model_name,
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        attention_mask = self.attention_mask_multiplier * attention_mask

        if self.quant_sim is None:
            return super().forward(input_tokens, attention_mask, *rest)

        layer_types = self._get_layer_types()
        text_config = self._get_text_config()
        linear_attn_config = self._get_linear_attn_config()
        num_full_attention = sum(1 for lt in layer_types if lt == "full_attention")

        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            position_ids = rest[0]
            state_tensors = rest[1:]
        else:
            position_ids_cos = rest[0]
            position_ids_sin = rest[1]
            state_tensors = rest[2:]

        kv_only_mode = len(state_tensors) == num_full_attention * 2

        if not kv_only_mode:
            return super().forward(input_tokens, attention_mask, *rest)

        if state_tensors and state_tensors[0].abs().sum() == 0:
            self._linear_attn_cache.clear()

        session = self.quant_sim.session
        onnx_input_names = [inp.name for inp in session.get_inputs()]
        onnx_output_names = [out.name for out in session.get_outputs()]
        input_dict: dict[str, torch.Tensor] = {}
        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            input_dict["input_ids"] = input_tokens
            input_dict["position_ids"] = position_ids
        elif self.llm_io_type == LLMIOType.genie_input_embeds:
            input_dict["inputs_embeds"] = input_tokens
            input_dict["position_ids_cos"] = position_ids_cos
            input_dict["position_ids_sin"] = position_ids_sin
        else:
            input_dict["input_ids"] = input_tokens
            input_dict["position_ids_cos"] = position_ids_cos
            input_dict["position_ids_sin"] = position_ids_sin
        input_dict["attention_mask"] = attention_mask

        kv_tensor_idx = 0
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                input_dict[f"past_key_{i}_in"] = state_tensors[kv_tensor_idx]
                input_dict[f"past_value_{i}_in"] = state_tensors[kv_tensor_idx + 1]
                kv_tensor_idx += 2
            else:
                if i in self._linear_attn_cache:
                    conv_state, recurrent_state = self._linear_attn_cache[i]
                    input_dict[f"conv_state_{i}_in"] = conv_state.to(input_tokens.device)
                    input_dict[f"recurrent_state_{i}_in"] = recurrent_state.to(input_tokens.device)
                else:
                    conv_kernel_dim = linear_attn_config["linear_conv_kernel_dim"]
                    key_dim = (
                        linear_attn_config["linear_num_key_heads"]
                        * linear_attn_config["linear_key_head_dim"]
                    )
                    value_dim = (
                        linear_attn_config["linear_num_value_heads"]
                        * linear_attn_config["linear_value_head_dim"]
                    )
                    conv_dim = key_dim * 2 + value_dim
                    num_v_heads = linear_attn_config["linear_num_value_heads"]
                    k_head_dim = linear_attn_config["linear_key_head_dim"]
                    v_head_dim = linear_attn_config["linear_value_head_dim"]

                    input_dict[f"conv_state_{i}_in"] = torch.zeros(
                        1, conv_dim, conv_kernel_dim,
                        device=input_tokens.device, dtype=torch.float32,
                    )
                    input_dict[f"recurrent_state_{i}_in"] = torch.zeros(
                        1, num_v_heads, k_head_dim, v_head_dim,
                        device=input_tokens.device, dtype=torch.float32,
                    )

        onnx_input_feed = {}
        for name in onnx_input_names:
            if name in input_dict:
                onnx_input_feed[name] = input_dict[name].cpu().detach().numpy()
            else:
                inp_info = next(inp for inp in session.get_inputs() if inp.name == name)
                shape = [d if isinstance(d, int) else 1 for d in inp_info.shape]
                onnx_input_feed[name] = torch.zeros(
                    shape,
                    dtype=torch.float32,
                ).numpy()

        output_np = session.run(None, onnx_input_feed)
        output_dict = dict(zip(onnx_output_names, output_np))

        def _get_output(name: str):
            if name in output_dict:
                return output_dict[name]
            updated_name = name + "_updated"
            if updated_name in output_dict:
                return output_dict[updated_name]
            raise KeyError(f"Output '{name}' not found in model outputs. Available: {list(output_dict.keys())}")

        result: list[torch.Tensor] = [torch.from_numpy(_get_output("logits"))]

        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                result.append(torch.from_numpy(_get_output(f"past_key_{i}_out")))
                result.append(torch.from_numpy(_get_output(f"past_value_{i}_out")))
            else:
                conv_out = torch.from_numpy(_get_output(f"conv_state_{i}_out")).detach()
                rec_out = torch.from_numpy(_get_output(f"recurrent_state_{i}_out")).detach()
                conv_kernel_dim = linear_attn_config["linear_conv_kernel_dim"]
                if conv_out.shape[2] > conv_kernel_dim:
                    conv_out = conv_out[:, :, -conv_kernel_dim:]
                self._linear_attn_cache[i] = (conv_out.cpu(), rec_out.cpu())

        return result

    @classmethod
    def get_onnx_export_input_spec(
        cls,
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec | None:
        return cls.get_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )

    @classmethod
    def get_onnx_export_output_names(
        cls,
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> list[str] | None:
        from qai_hub_models.models.qwen3_5_0_8b.model import LAYER_TYPES
        return cls._get_output_names(
            num_hidden_layers=llm_config.get("num_hidden_layers", 24),
            layer_types=LAYER_TYPES,
        )

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Make sure AIMET encodings are ready for ONNX split."""
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        assert uses_lists

        # Convert encodings to dictionaries for faster look-ups
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings["activation_encodings"]
        }
        encodings["param_encodings"] = {
            v["name"]: v for v in encodings["param_encodings"]
        }

        # Propagate embedding encodings
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        encodings["activation_encodings"][embed_a_name] = copy.deepcopy(
            encodings["activation_encodings"][embed_w_name]
        )
        for key in encodings["activation_encodings"]:
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        # Convert back
        encodings["activation_encodings"] = list(
            encodings["activation_encodings"].values()
        )
        encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


class Qwen3_5Base_QNN(LLM_QNN):
    state_schema_mode = StateSchemaMode.HYBRID
    FPModel = Qwen3_5Base
    EmbeddingClass = Qwen3_5RopeEmbedding
    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds
    split_embedding = False
    num_layers_per_split: int

    @staticmethod
    def _get_output_names(
        num_hidden_layers: int,
        layer_types: list[str] | None = None,
    ) -> list[str]:
        output_names = ["logits"]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                output_names.append(f"past_key_{i}_out")
                output_names.append(f"past_value_{i}_out")
            else:
                output_names.append(f"conv_state_{i}_out")
                output_names.append(f"recurrent_state_{i}_out")
        return output_names
