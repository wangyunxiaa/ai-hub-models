# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Items defined in this file require that AIMET-ONNX be installed."""

from __future__ import annotations

from packaging import version

from qai_hub_models.utils.base_model import BaseModel

try:
    import aimet_onnx
    from aimet_onnx.common.utils import AimetLogger
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

    aimet_onnx_is_installed = True
except (ImportError, ModuleNotFoundError):
    aimet_onnx_is_installed = False
import contextlib
import importlib.metadata
import itertools
import os
import shutil
import sys
from collections.abc import Collection, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import onnxruntime
import torch
from qai_hub.client import DatasetEntries
from tqdm.autonotebook import tqdm

from qai_hub_models.evaluators.base_evaluators import _DataLoader
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.protocols import PretrainedHubModelProtocol
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, qaihm_temp_dir
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.dataset_util import DataLoader, dataset_entries_to_dataloader
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import mock_torch_onnx_inference
from qai_hub_models.utils.runtime_torch_wrapper import kwargs_to_dict

DEFAULT_SEQ_MSE_NUM_SAMPLES = 20
DEFAULT_ADA_SCALE_NUM_SAMPLES = 128
DEFAULT_ADA_SCALE_NUM_ITERATIONS = 512
DEFAULT_SPIN_QUANT_NUM_ITERATIONS = 200


def ensure_aimet_onnx_installed(
    expected_version: str | None = None, model_id: str | None = None
) -> None:
    if not aimet_onnx_is_installed:
        errstr = "AIMET-ONNX is missing but must be installed. "
        if not sys.platform.startswith("linux") and sys.platform not in [
            "win32",
            "cygwin",
        ]:
            errstr += "It is not supported on this operating system. You must use either Linux or Windows Subsystem for Linux to install AIMET-ONNX."
        else:
            if model_id is not None:
                install_target = f'"qai_hub_models[{model_id}]"'
            elif expected_version is not None:
                install_target = f"aimet-onnx=={expected_version}"
            else:
                install_target = '"qai_hub_models[<your_target_model_id_here>]"'

            if sys.platform in ["win32", "cygwin"]:
                errstr += "AIMET-ONNX is not supported on Windows. We suggest using Windows Subsystem for Linux (WSL) to create a python environment compatible with AIMET-ONNX.\nIn a compatible WSL python env, run "
            else:
                errstr += "Run "
            errstr += f"`pip install {install_target}` to install the correct version of AIMET-ONNX."

        if model_id is not None:
            errstr += f"\nAlternatively, for model export, you may run `python -m qai_hub_models.models.{model_id}.export.py --fetch-static-assets` to fetch pre-compiled assets for this model."

        raise RuntimeError(errstr)


def ensure_min_aimet_onnx_version(
    expected_version: str, model_id: str | None = None
) -> None:
    ensure_aimet_onnx_installed(expected_version, model_id)
    if version.Version(aimet_onnx.__version__) < version.Version(expected_version):
        raise RuntimeError(
            f"Installed AIMET-ONNX version not supported. Expected >= {expected_version}, got {aimet_onnx.__version__!s}\n"
            f"Please run `pip install aimet-onnx=={expected_version}`"
        )


def ensure_max_aimet_onnx_version(
    expected_version: str, model_id: str | None = None
) -> None:
    ensure_aimet_onnx_installed(expected_version, model_id)
    if version.Version(aimet_onnx.__version__) < version.Version(expected_version):
        raise RuntimeError(
            f"Installed AIMET-ONNX version not supported. Expected=<{expected_version}, got {aimet_onnx.__version__!s}\n"
            f"Please run `pip install transformers=={expected_version}`"
        )


def _fix_unsupported_channel_axis(quant_sim: QuantSimOnnx) -> None:
    for name, quantizer in quant_sim.qc_quantize_op_dict.items():
        if not quantizer.enabled:
            continue
        if not quantizer.quant_info.usePerChannelMode:
            continue
        channel_axis = quantizer.quant_info.channelAxis
        if channel_axis not in (0, 1):
            if (
                quantizer.tensor_quantizer_params is not None
                and quantizer.tensor_quantizer_params.tensor_shape is not None
            ):
                tensor_shape = quantizer.tensor_quantizer_params.tensor_shape
                ndim = len(tensor_shape)
                print(
                    f"[SeqMSE Fix] Quantizer '{name}' has unsupported channel_axis={channel_axis}, "
                    f"tensor_shape={tensor_shape}, blockSize={quantizer.quant_info.blockSize}, "
                    f"blockAxis={quantizer.quant_info.blockAxis}"
                )
                if ndim == 4 and channel_axis == ndim - 1:
                    quantizer.quant_info.channelAxis = 0
                    quantizer.quant_info.blockAxis = 1
                    quantizer._tensor_quantizer = quantizer._build_tensor_quantizer()
                    print(
                        f"[SeqMSE Fix] Fixed quantizer '{name}': channelAxis=0, blockAxis=1"
                    )
                elif ndim == 4 and channel_axis == ndim - 2:
                    quantizer.quant_info.channelAxis = 1
                    quantizer.quant_info.blockAxis = 0
                    quantizer._tensor_quantizer = quantizer._build_tensor_quantizer()
                    print(
                        f"[SeqMSE Fix] Fixed quantizer '{name}': channelAxis=1, blockAxis=0"
                    )
                else:
                    print(
                        f"[SeqMSE Fix] WARNING: Cannot auto-fix quantizer '{name}' with "
                        f"channel_axis={channel_axis}, ndim={ndim}. "
                        f"Disabling per-channel mode for this quantizer."
                    )
                    quantizer.quant_info.usePerChannelMode = False
                    quantizer.quant_info.channelAxis = 0
                    quantizer.quant_info.blockSize = 0
                    quantizer._tensor_quantizer = quantizer._build_tensor_quantizer()


_trilu_converter_registered = False


def _register_trilu_converter() -> None:
    global _trilu_converter_registered
    if _trilu_converter_registered:
        return
    try:
        from onnx2torch.node_converters.registry import (
            OperationDescription,
            _CONVERTER_REGISTRY,
            add_converter,
        )
        from onnx2torch.onnx_graph import OnnxGraph
        from onnx2torch.onnx_node import OnnxNode
        from onnx2torch.utils.common import (
            OnnxToTorchModule,
            OperationConverterResult,
            onnx_mapping_from_node,
        )
    except ImportError:
        return

    trilu_desc = OperationDescription(
        domain="",
        operation_type="Trilu",
        version=14,
    )
    if trilu_desc in _CONVERTER_REGISTRY:
        _trilu_converter_registered = True
        return

    import torch
    from torch import nn

    class OnnxTrilu(nn.Module, OnnxToTorchModule):
        def __init__(self, upper: bool = True):
            super().__init__()
            self.upper = upper

        def forward(
            self, input_tensor: torch.Tensor, diagonal: torch.Tensor | None = None
        ) -> torch.Tensor:
            diag_val = 0
            if diagonal is not None:
                diag_val = int(diagonal.item())
            if self.upper:
                return torch.triu(input_tensor, diagonal=diag_val)
            return torch.tril(input_tensor, diagonal=diag_val)

    @add_converter(operation_type="Trilu", version=14)
    def _(
        node: OnnxNode, graph: OnnxGraph
    ) -> OperationConverterResult:
        upper = node.attributes.get("upper", 1) != 0
        return OperationConverterResult(
            torch_module=OnnxTrilu(upper=upper),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    _trilu_converter_registered = True


def _find_block_state_inputs(sim_model, block_input_name, block_output_name, all_state_names, common_input_names=None):
    import onnx_ir

    graph = sim_model.graph
    values = onnx_ir.convenience.create_value_mapping(graph, include_subgraphs=False)

    output_val = values[block_output_name]
    input_frontier = set()
    visited_nodes = set()
    visited_values = set()
    value_stack = [output_val]

    boundary_names = {block_input_name}
    if common_input_names:
        boundary_names.update(common_input_names)

    while value_stack:
        value = value_stack.pop()
        if value in visited_values:
            continue
        visited_values.add(value)
        if value.is_initializer():
            continue
        if value.name in boundary_names:
            continue
        producer = value.producer()
        if producer is not None and producer not in visited_nodes:
            visited_nodes.add(producer)
            for inp in producer.inputs:
                if inp is not None and inp not in visited_values:
                    value_stack.append(inp)

    for node in visited_nodes:
        for inp in node.inputs:
            if inp is None:
                continue
            producer = inp.producer()
            if producer is None or producer not in visited_nodes:
                input_frontier.add(inp)

    state_name_set = set(all_state_names)
    block_state_names = []
    for val in sorted(input_frontier, key=lambda v: v.name or ""):
        if val.name in state_name_set:
            block_state_names.append(val.name)

    return block_state_names


def _qwen3_5_apply_adascale(cls, sim, inputs, adascale_model_config, num_iterations):
    import gc
    import tempfile

    import onnx_ir
    import torch

    from aimet_onnx.experimental.adascale.adascale_optimizer import (
        _DEBUG_NUM_PARTIAL_ITERATIONS,
        _DEBUG_NUM_PARTIAL_ITERATIONS_END,
        _DEBUG_NUM_PARTIAL_ITERATIONS_START,
        AdaScale,
        get_decoder_blocks_end_points,
    )
    from aimet_onnx.experimental.adascale.activation_sampler import ActivationSampler
    from aimet_onnx.utils import get_torch_device
    from aimet_onnx import ir_utils

    _orig_optimize_adascale_block = AdaScale.optimize_adascale_block

    @staticmethod
    def _patched_optimize_adascale_block(
        sim_model,
        quantizer_dict,
        fp_inputs,
        quantized_inputs,
        block_input_output_names,
        beta_gamma_lr=1e-3,
        scales_lr=5e-4,
        num_iterations=1500,
        device=torch.device("cpu"),
    ):
        from aimet_onnx.experimental.adascale.model_converter import (
            copy_pt_encodings_to_sim,
            copy_pt_weights_to_onnx,
            get_pt_block,
        )
        from aimet_onnx.experimental.adascale.utils import (
            change_tensor_device_placement,
            convert_to_torch,
        )
        from aimet_onnx.experimental.adascale.quantizer import (
            add_qlinear_layers,
            get_adascale_trainable_params,
            replace_with_adascale_quantizers,
        )

        pytorch_block, pt_weights_to_onnx_initializers = get_pt_block(
            sim_model, block_input_output_names
        )
        pytorch_block.requires_grad_(False)

        torch_fp_input = convert_to_torch(fp_inputs)
        torch_quant_input = convert_to_torch(quantized_inputs)

        gc.collect()
        torch.cuda.empty_cache()

        pytorch_block.to(device)
        fp_out = []
        with torch.no_grad():
            for input_tensor in torch_fp_input:
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = [input_tensor]

                input_tensor = [
                    inp_t.to(device=device) for inp_t in input_tensor
                ]
                out = pytorch_block(*input_tensor).detach()

                out.requires_grad_(False)
                fp_out.append(change_tensor_device_placement(out, torch.device("cpu")))
                del out, input_tensor
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        pytorch_block = add_qlinear_layers(
            pytorch_block, bitwidth=AdaScale.ADASCALE_PARAM_BW
        )
        replace_with_adascale_quantizers(pytorch_block)

        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            pytorch_block
        )
        adascale_params = all_beta_gamma_parameters + all_scale_parameters
        for p in adascale_params:
            p.requires_grad = True

        trainable_params = [
            {
                "params": all_beta_gamma_parameters,
                "lr": beta_gamma_lr,
            },
            {
                "params": all_scale_parameters,
                "lr": scales_lr,
            },
        ]

        optimizer = torch.optim.Adam(trainable_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations, eta_min=0.0
        )

        gc.collect()
        torch.cuda.empty_cache()

        pytorch_block.to(device)
        with torch.set_grad_enabled(True):
            for iteration in tqdm(range(num_iterations)):
                fp_input = torch_fp_input[iteration % len(torch_fp_input)]
                quant_input = torch_quant_input[iteration % len(torch_quant_input)]
                input_tensor = quant_input
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = [input_tensor]
                input_tensor = [
                    inp_t.to(device=device) for inp_t in input_tensor
                ]
                quant_out = pytorch_block(*input_tensor)
                batch_fp_out = fp_out[iteration % len(torch_fp_input)].to(device)
                loss = torch.nn.functional.mse_loss(quant_out, batch_fp_out)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                del quant_out, batch_fp_out, loss, input_tensor, fp_input, quant_input

                if iteration % 10 == 0:
                    torch.cuda.empty_cache()

        copy_pt_weights_to_onnx(
            pytorch_block, sim_model, pt_weights_to_onnx_initializers
        )
        copy_pt_encodings_to_sim(
            pytorch_block, quantizer_dict, pt_weights_to_onnx_initializers
        )

        del (
            pytorch_block,
            torch_quant_input,
            torch_fp_input,
            optimizer,
            pt_weights_to_onnx_initializers,
            fp_out,
            fp_inputs,
            quantized_inputs,
        )
        gc.collect()
        torch.cuda.empty_cache()

    AdaScale.optimize_adascale_block = _patched_optimize_adascale_block

    try:
        with cls._disable_activation_quantizers(sim):
            sim._compute_param_encodings(overwrite=False)

            blocks_end_points = get_decoder_blocks_end_points(
                sim, adascale_model_config.model_type
            )

            device = get_torch_device(sim.session)
            graph_input_names = [inp.name for inp in sim.session.get_inputs()]
            if graph_input_names != list(inputs[0].keys()):
                raise ValueError(
                    "Graph input names do not match the keys in the provided inputs."
                )

            print("graph_input_names: ", graph_input_names)
            common_input_names = []
            for name in graph_input_names:
                if "attention" in name:
                    common_input_names.append(name)
                if "position" in name:
                    common_input_names.append(name)
                if "input_ids" in name or "input_embeds" in name:
                    common_input_names.append(name)

            all_state_names = [
                name
                for name in graph_input_names
                if name.startswith("past_key_")
                or name.startswith("past_value_")
                or name.startswith("conv_state_")
                or name.startswith("recurrent_state_")
            ]

            del sim.session
            gc.collect()
            torch.cuda.empty_cache()

            with tempfile.TemporaryDirectory() as tempdir:
                fp32_path = f"{tempdir}/fp32_model.onnx"
                sim_path = f"{tempdir}/sim_model.onnx"
                sim_model = onnx_ir.from_proto(sim.model.model)
                onnx_ir.passes.common.TopologicalSortPass().call(sim_model)
                fp32_model = sim_model.clone()
                ir_utils.remove_aimet_quantizers(fp32_model)
                onnx_ir.save(fp32_model, fp32_path, external_data="fp32_model.data")

                del fp32_model
                gc.collect()
                torch.cuda.empty_cache()

                for idx in range(len(blocks_end_points)):
                    if (
                        _DEBUG_NUM_PARTIAL_ITERATIONS is not None
                        and idx >= _DEBUG_NUM_PARTIAL_ITERATIONS
                    ):
                        break
                    if (
                        _DEBUG_NUM_PARTIAL_ITERATIONS_START is not None
                        and _DEBUG_NUM_PARTIAL_ITERATIONS_END is not None
                        and (
                            idx < _DEBUG_NUM_PARTIAL_ITERATIONS_START
                            or idx >= _DEBUG_NUM_PARTIAL_ITERATIONS_END
                        )
                    ):
                        continue

                    block_input_name = blocks_end_points[idx][0].inputs[0].name
                    block_output_name = blocks_end_points[idx][1].inputs[0].name

                    block_state_tensor_names = _find_block_state_inputs(
                        sim_model, block_input_name, block_output_name, all_state_names, common_input_names
                    )
                    print(f"idx: {idx}, block_state_tensor_names: {block_state_tensor_names}")

                    block_input_names = list(common_input_names)
                    if len(block_state_tensor_names) > 0:
                        block_input_names.extend(block_state_tensor_names)

                    gc.collect()
                    torch.cuda.empty_cache()

                    onnx_ir.save(sim_model, path=sim_path, external_data="sim_model.data")
                    qsim_sess = ActivationSampler(
                        blocks_end_points[idx][0].inputs[0].name,
                        sim_path,
                        sim.providers,
                    )

                    fp_inputs, qsim_inputs = [], []
                    for input_data in inputs:
                        qsim_inputs.append(qsim_sess.sample_acts(input_data))
                    del qsim_sess
                    gc.collect()
                    torch.cuda.empty_cache()

                    fp32_sampler = ActivationSampler(
                        blocks_end_points[idx][0].inputs[0].name,
                        fp32_path,
                        sim.providers,
                    )
                    for input_data in inputs:
                        fp_inputs.append(fp32_sampler.sample_acts(input_data))
                    del fp32_sampler
                    gc.collect()
                    torch.cuda.empty_cache()

                    fp_input_list = []
                    qsim_input_list = []
                    for i in range(len(fp_inputs)):
                        fp_list, qsim_list = [], []
                        fp_list.append(fp_inputs[i])
                        qsim_list.append(qsim_inputs[i])
                        for name in block_input_names:
                            fp_list.append(inputs[i][name])
                            qsim_list.append(inputs[i][name])
                        fp_input_list.append(fp_list)
                        qsim_input_list.append(qsim_list)

                    block_input_output_names = AdaScale.get_block_start_end_name(
                        blocks_end_points, idx, block_input_names
                    )
                    print("idx: ", idx, "block_input_output_names: ", block_input_output_names)

                    gc.collect()
                    torch.cuda.empty_cache()

                    AdaScale.optimize_adascale_block(
                        sim_model,
                        sim.qc_quantize_op_dict,
                        fp_input_list,
                        qsim_input_list,
                        block_input_output_names,
                        adascale_model_config.beta_gamma_lr,
                        adascale_model_config.scales_lr,
                        num_iterations,
                        device,
                    )
                    del fp_input_list, qsim_input_list, fp_inputs, qsim_inputs
                    gc.collect()
                    torch.cuda.empty_cache()

                sim.model.model.CopyFrom(onnx_ir.to_proto(sim_model))
                sim._rebuild_session()
    finally:
        AdaScale.optimize_adascale_block = _orig_optimize_adascale_block


@contextmanager
def set_aimet_log_level(log_level: int) -> Generator[None, None, None]:
    area_log_levels: dict[AimetLogger.LogAreas, int] = {}
    for area in AimetLogger.LogAreas:
        area_log_levels[area] = AimetLogger.get_area_logger(area).level

    try:
        AimetLogger.set_level_for_all_areas(log_level)
        yield
    finally:
        for area, level in area_log_levels.items():
            AimetLogger.set_area_logger_level(area, level)


class AIMETOnnxQuantizableMixin(PretrainedHubModelProtocol):
    """
    Mixin that allows a model to be quantized & exported to disk using AIMET.
    Inheritor must implement BaseModel for this mixin to function.
    """

    # For pre-calibrated asset lookup
    model_id: str = ""
    model_asset_version: int = -1

    # Which AIMET model type to use for AdaScale
    # (if None, the model cannot use AdaScale)
    ada_scale_model_type: str | None = None

    # RMSNorms per block (currently needed for AdaScale for Qwen3)
    ada_scale_num_rmsnorm_per_blk: int | None = None

    def __init__(
        self,
        quant_sim: QuantSimOnnx | None,
    ) -> None:
        self.quant_sim = quant_sim
        if self.quant_sim is not None:
            self.input_names = [i.name for i in self.quant_sim.session.get_inputs()]
            self.output_names = [
                output.name for output in self.quant_sim.session.get_outputs()
            ]

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        # This must be defined by the PretrainedHubModelProtocol ABC
        raise ValueError(
            f"Cannot call convert_to_torchscript on {self.__class__.__name__}"
        )

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        """
        Get calibration data for quantization.

        Parameters
        ----------
        input_spec
            The input specification for the model.
        num_samples
            None to use all. Specify `num_samples` to use fewer. If
            `num_samples` are more than available, use all available (same
            behavior as None).

        Returns
        -------
        calibration_data : DatasetEntries | None
            The calibration dataset entries, or None if not available.
        """
        return None

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        """
        Get the calibrated AIMET model paths.

        Returns
        -------
        onnx_path : str
            Path to .onnx file.
        encodings_path : str
            Path to .encodings file.
        """
        if not cls.model_id or cls.model_asset_version == -1:
            raise ValueError("model_id and model_asset_version must be defined")

        subfolder = Path(getattr(cls, "default_subfolder", ""))

        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            cls.model_id,
            cls.model_asset_version,
            str(subfolder / "model.onnx"),
        ).fetch()
        with contextlib.suppress(Exception):
            _ = CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                str(subfolder / "model.data"),
            ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            cls.model_id,
            cls.model_asset_version,
            str(subfolder / "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings

    def _dataloader_to_numpy(
        self, data: _DataLoader, num_batches: int
    ) -> list[dict[str, Any]]:
        assert self.quant_sim is not None
        input_names = [inp.name for inp in self.quant_sim.session.get_inputs()]
        onnx_data = []
        n = min(len(data), num_batches)
        for batch in tqdm(itertools.islice(data, n), total=n):
            onnx_data.append(  # noqa: PERF401
                {
                    k: v.cpu().detach().numpy()
                    for k, v in kwargs_to_dict(input_names, *batch).items()
                }
            )
        return onnx_data

    def _apply_seq_mse(self, data: _DataLoader, num_batches: int) -> None:
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.8.0")
        _fix_unsupported_channel_axis(self.quant_sim)
        aimet_onnx.apply_seq_mse(
            self.quant_sim, self._dataloader_to_numpy(data, num_batches)
        )

    def _apply_ada_scale(
        self,
        data: _DataLoader,
        num_batches: int,
        num_iterations: int,
        model_type: str,
        num_rmsnorm_per_blk: int | None = None,
    ) -> None:
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.26.0")
        from aimet_onnx.experimental.adascale.adascale_optimizer import (
            AdaScale,
            AdaScaleModelConfig,
            adascale_model_config_dict,
        )

        _register_trilu_converter()

        if model_type == "qwen3_5":
            if "qwen3_5" not in adascale_model_config_dict:
                adascale_model_config_dict["qwen3_5"] = AdaScaleModelConfig(
                    model_type="qwen3_5",
                    beta_gamma_lr=1e-3,
                    scales_lr=5e-4,
                )
            import aimet_onnx.experimental.adascale.find_blocks as find_blocks_mod
            import aimet_onnx.experimental.adascale.adascale_optimizer as adascale_mod
            _orig_get_decoder_blocks = find_blocks_mod.get_decoder_blocks_end_points

            def _patched_get_decoder_blocks(quantsim, mt):
                if mt == "qwen3_5":
                    mt = "qwen3"
                return _orig_get_decoder_blocks(quantsim, mt)

            find_blocks_mod.get_decoder_blocks_end_points = _patched_get_decoder_blocks
            adascale_mod.get_decoder_blocks_end_points = _patched_get_decoder_blocks

            _orig_apply_adascale = AdaScale.apply_adascale

            @classmethod
            def _patched_apply_adascale(cls, sim, inputs, adascale_model_config, num_iterations=1500):
                return _qwen3_5_apply_adascale(
                    cls, sim, inputs, adascale_model_config, num_iterations
                )

            AdaScale.apply_adascale = _patched_apply_adascale

        restore_value: int | None = None
        if model_type in ("qwen3", "qwen3_5") and num_rmsnorm_per_blk is not None:
            from aimet_onnx.graph_passes.passes.decoder_block import DecoderBlockQwen3

            restore_value = DecoderBlockQwen3.NUM_RMSNORM_PER_BLK
            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = num_rmsnorm_per_blk

        AdaScale.apply_adascale(
            self.quant_sim,
            self._dataloader_to_numpy(data, num_batches=num_batches),
            adascale_model_config=adascale_model_config_dict[model_type],
            num_iterations=num_iterations,
        )

        if model_type == "qwen3_5":
            AdaScale.apply_adascale = _orig_apply_adascale

        if restore_value is not None:
            DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = restore_value

    def _apply_spin_quant(
        self,
        data: _DataLoader,
        num_batches: int,
        num_iterations: int = 200,
    ) -> None:
        assert self.quant_sim is not None
        from aimet_onnx.experimental.spinquant import apply_spinquant

        embedding = None
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            fp_model = self.FPModel.from_pretrained.__func__
            from qai_hub_models.models._shared.llm.model import LLMBase
            fp_instance = None
            try:
                fp_cls = self.FPModel
                fp_instance = fp_cls.__new__(fp_cls)
                embedding_layer = fp_instance.model.get_input_embeddings() if hasattr(fp_instance, 'model') else None
            except Exception:
                pass

        print("SpinQuant: Applying Hadamard rotations via AIMET apply_spinquant")
        apply_spinquant(self.quant_sim, embedding=embedding)
        print("SpinQuant: Rotations applied successfully")

    def _apply_calibration(self, data: DataLoader, num_batches: int) -> None:
        assert self.quant_sim is not None
        ensure_min_aimet_onnx_version("2.8.0")
        self.quant_sim.compute_encodings(self._dataloader_to_numpy(data, num_batches))

    def quantize(
        self,
        data: DataLoader | None = None,
        num_samples: int | None = None,
        use_seq_mse: bool = False,
        use_ada_scale: bool = False,
        use_spin_quant: bool = False,
        seq_mse_num_samples: int | None = None,
        ada_scale_num_samples: int | None = None,
        ada_scale_num_iterations: int | None = None,
        spin_quant_num_samples: int | None = None,
        spin_quant_num_iterations: int | None = None,
    ) -> None:
        """
        Quantize the model using calibration data.

        Parameters
        ----------
        data
            If None, create data loader from get_calibration_data(), which
            must be implemented.
        num_samples
            Number of samples to use for calibration. If None, uses all
            available samples in the data loader.
        use_seq_mse
            Whether to apply sequential MSE optimization during quantization.
        use_ada_scale
            Whether to apply AdaScale optimization during quantization.
        use_spin_quant
            Whether to apply SpinQuant (learned rotation) optimization during quantization.
        seq_mse_num_samples
            Number of samples for sequential MSE. Defaults to num_samples.
        ada_scale_num_samples
            Number of samples for AdaScale.
        ada_scale_num_iterations
            Number of iterations for AdaScale.
        spin_quant_num_samples
            Number of samples for SpinQuant.
        spin_quant_num_iterations
            Number of iterations for SpinQuant.

        Returns
        -------
        None
        """
        if use_ada_scale and self.ada_scale_model_type is None:
            raise ValueError("AdaScale is not supported for this model.")

        if data is None:
            calib_data = self.get_calibration_data()
            if calib_data is None:
                raise ValueError(
                    "`data` must be specified if get_calibration_data is not defined."
                )
            data = dataset_entries_to_dataloader(calib_data)

        # "samples": 4096 context length batches
        # "batches": actual iterations
        if hasattr(self, "context_length") and hasattr(self, "sequence_length"):
            batches_per_sample = self.context_length // self.sequence_length
        else:
            batches_per_sample = 1

        if use_seq_mse:
            seq_mse_num_samples = min(
                len(data) // batches_per_sample,
                seq_mse_num_samples or num_samples or DEFAULT_SEQ_MSE_NUM_SAMPLES,
            )
            seq_mse_num_batches = seq_mse_num_samples * batches_per_sample

            print()
            print(
                f"Apply Sequential MSE ({seq_mse_num_samples} samples / {seq_mse_num_batches} batches)"
            )
            print()
            self._apply_seq_mse(data=data, num_batches=seq_mse_num_batches)

        if use_spin_quant:
            spin_quant_num_samples_val = min(
                len(data) // batches_per_sample,
                spin_quant_num_samples or num_samples or DEFAULT_SEQ_MSE_NUM_SAMPLES,
            )
            spin_quant_num_batches = spin_quant_num_samples_val * batches_per_sample
            spin_quant_num_iters = (
                spin_quant_num_iterations or DEFAULT_SPIN_QUANT_NUM_ITERATIONS
            )
            print()
            print(
                f"Apply SpinQuant ({spin_quant_num_samples_val} samples / {spin_quant_num_batches} batches, {spin_quant_num_iters} iterations)"
            )
            print()
            self._apply_spin_quant(
                data=data,
                num_batches=spin_quant_num_batches,
                num_iterations=spin_quant_num_iters,
            )

        if use_ada_scale:
            assert self.ada_scale_model_type is not None
            ada_scale_num_samples = min(
                len(data) // batches_per_sample,
                ada_scale_num_samples or DEFAULT_ADA_SCALE_NUM_SAMPLES,
            )
            ada_scale_num_iters = (
                ada_scale_num_iterations or DEFAULT_ADA_SCALE_NUM_ITERATIONS
            )
            ada_scale_num_batches = ada_scale_num_samples * batches_per_sample
            print()
            print(
                f"Apply AdaScale ({ada_scale_num_samples} samples / {ada_scale_num_batches} batches, {ada_scale_num_iters} iterations)"
            )
            print()
            self._apply_ada_scale(
                data=data,
                num_batches=ada_scale_num_batches,
                num_iterations=ada_scale_num_iters,
                model_type=self.ada_scale_model_type,
                num_rmsnorm_per_blk=self.ada_scale_num_rmsnorm_per_blk,
            )

        num_calib_samples = num_samples or len(data)
        num_calib_batches = num_calib_samples * batches_per_sample

        print()
        print(
            f"Start QuantSim calibration for {self.__class__.__name__} ({num_calib_samples} samples / {num_calib_batches} batches)"
        )
        print()
        self._apply_calibration(data=data, num_batches=num_calib_batches)

    @contextlib.contextmanager
    def remove_quantization(self) -> Generator[None, None, None]:
        """
        Context manager to temporarily remove quantization nodes from the model. Useful for prefilling data without
        quantization, e.g. for AdaScale or SeqMSE.
        """
        assert isinstance(self.quant_sim, QuantSimOnnx)
        with self.quant_sim._remove_quantization_nodes():
            self.quant_sim._rebuild_session()

            yield

        self.quant_sim._rebuild_session()

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        data = self.get_calibration_data()
        if data is None:
            # Fallback to BaseModel's impl
            data = BaseModel._sample_inputs_impl(cast(BaseModel, self), input_spec)
        assert isinstance(data, dict)
        return data

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """QuantSim forward pass with torch.Tensor"""
        assert self.quant_sim is not None
        return mock_torch_onnx_inference(self.quant_sim.session, *args, **kwargs)

    def save_calibrated_checkpoint(self, output_checkpoint: str) -> None:
        """Save AIMET-ONNX checkpoint to output_checkpoint/subfolder, if"""
        default_subfolder = getattr(self.__class__, "default_subfolder", "")
        export_dir = output_checkpoint
        if default_subfolder:
            export_dir = str(Path(output_checkpoint) / default_subfolder)

        shutil.rmtree(export_dir, ignore_errors=True)
        os.makedirs(export_dir, exist_ok=True)

        print(f"Saving quantized {self.__class__.__name__} to {export_dir}")
        assert self.quant_sim is not None
        self.quant_sim.export(str(export_dir), "model")
        print(f"{self.__class__.__name__} saved to {export_dir}")

    @staticmethod
    def get_ort_providers(
        device: torch.device,
    ) -> list[str | tuple[str, dict[str, int]]]:
        if device.type == "cuda":
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                msg = (
                    f"WARNING: GPU requested but CUDAExecutionProvider is not available. "
                    f"Falling back to CPU. Available providers: {available}"
                )
                ort_packages = [
                    d.name
                    for d in importlib.metadata.distributions()
                    if d.name and d.name.startswith("onnxruntime")
                ]
                if "onnxruntime" in ort_packages and any(
                    p != "onnxruntime" for p in ort_packages
                ):
                    msg += (
                        f"\nThis may be caused by the 'onnxruntime' (CPU) package "
                        f"shadowing a GPU-enabled variant. "
                        f"Installed onnxruntime packages: {ort_packages}. "
                        f"Try: pip uninstall onnxruntime && pip install onnxruntime-gpu"
                    )
                print(msg)
                return ["CPUExecutionProvider"]
            return (
                [
                    ("CUDAExecutionProvider", {"device_id": device.index}),
                    "CPUExecutionProvider",
                ]
                if device.index is not None
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        return ["CPUExecutionProvider"]

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        model_name: str | None = None,
        return_zip: bool = True,
    ) -> str:
        """
        Converts the torch module to a zip file containing an unquantized ONNX model
        and an AIMET quantization encodings file if return_zip is True (default).

        If return_zip is False, the model is exported to a directory.
        In that case, the output directory is set to:

            Path(output_dir) / f"{model_name}.aimet"

        and the existing directory is forcefully removed.
        """
        if model_name is None:
            model_name = self.__class__.__name__

        output_dir = Path(output_dir)

        if return_zip:
            # Ensure output_dir exists and define the zip path.
            os.makedirs(output_dir, exist_ok=True)
            zip_path = output_dir / f"{model_name}.aimet.zip"
            base_dir = Path(f"{model_name}.aimet")

            print(f"Exporting quantized {self.__class__.__name__} to {zip_path}")
            # Use a temporary directory to export the model before zipping.
            with qaihm_temp_dir() as tmpdir:
                export_dir = Path(tmpdir) / base_dir
                os.makedirs(export_dir)
                assert self.quant_sim is not None
                self.quant_sim.export(str(export_dir), "model")

                onnx_file_path = str(export_dir / "model.onnx")
                encoding_file_path = str(export_dir / "model.encodings")

                # Attempt to locate external data file.
                # aimet-onnx<=2.0.0 export external data with model.onnx.data
                # aimet-onnx>=2.3.0 export external data with model.data
                # version between 2.0 - 2.3 are broken on large models
                external_data_file_path = ""
                external_data_file_path2 = export_dir / "model.onnx.data"
                external_data_file_path1 = export_dir / "model.data"
                if external_data_file_path1.exists():
                    external_data_file_path = str(external_data_file_path1)
                elif external_data_file_path2.exists():
                    external_data_file_path = str(external_data_file_path2)

                zip_aimet_model(
                    str(zip_path),
                    base_dir,
                    onnx_file_path,
                    encoding_file_path,
                    external_data_file_path,
                )
            return str(zip_path)
        # Export directly to a directory at output_dir / f"{model_name}.aimet"
        export_dir = output_dir / f"{model_name}.aimet"
        shutil.rmtree(export_dir, ignore_errors=True)
        os.makedirs(export_dir, exist_ok=True)

        print(
            f"Exporting quantized {self.__class__.__name__} to directory {export_dir}"
        )
        assert self.quant_sim is not None
        self.quant_sim.export(str(export_dir), "model")
        return str(export_dir)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        """AI Hub Workbench quantize options recommended for the model."""
        return other_options or ""
