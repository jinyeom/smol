from pathlib import Path
from typing import Union, Sequence, Optional, Tuple
import torch
from torch import nn
import onnx
from onnxsim import simplify as onnx_simplify


def export_state_dict(model: nn.Module, path: Union[str, Path]) -> Path:
    model = model.eval().cpu()
    path = Path(path)
    torch.save(model.state_dict(), path)
    return path


def export_onnx(
    model: nn.Module,
    path: Union[str, Path],
    input_shape: Tuple[int, int, int, int],
    input_names: Optional[Sequence[int]] = None,
    output_names: Optional[Sequence[int]] = None,
    opset_version: int = 10,
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
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        verbose=verbose,
        keep_initializers_as_inputs=True,
    )
    if simplify:
        simple_onnx, success = onnx_simplify(str(path), perform_optimization=True)
        assert success, "failed to simplify the exported ONNX file"
        onnx.save(simple_onnx, str(path))
    return path
