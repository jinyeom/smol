from pathlib import Path
from typing import Union, Sequence, Optional, Tuple
import torch
from torch import nn
import onnx

# from onnxsim import simplify as onnx_simplify
from smol.utils.onnx_simplifier import simplify as onnx_simplify


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
    path = Path(path)
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
        simple_onnx, success = onnx_simplify(str(path))
        assert success, "failed to simplify the exported ONNX file"
        onnx.save(simple_onnx, str(path))
    return path
