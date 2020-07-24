from pathlib import Path
from typing import Union, Sequence, Optional, Tuple
import torch
from torch import nn
import onnx

from smol.utils.onnx import simplify as onnx_simplify


def export_state_dict(model: nn.Module, path: Union[str, Path]) -> Path:
    model = model.eval().cpu()
    path = Path(path)
    torch.save(model.state_dict(), path)
    return path


def export_onnx(
    model: nn.Module,
    path: Union[str, Path],
    input_shape: Tuple[int, int, int, int],
    opset_version: int = 11,
    verbose: bool = False,
    simplify: bool = True,
) -> Path:
    model = model.eval().cpu()
    dummy_input = torch.rand(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        do_constant_folding=True,
        keep_initializers_as_inputs=True,
    )
    if simplify:
        onnx_model = onnx.load(path)
        onnx_model = onnx_simplify(onnx_model)
        onnx.save(onnx_model, path)
    return path
