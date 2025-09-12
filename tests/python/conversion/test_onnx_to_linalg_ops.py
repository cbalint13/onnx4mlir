import pytest
import numpy as np
import os
import sys
import difflib
import textwrap
from mlir.dialects import func
from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    RankedTensorType,
    DenseElementsAttr,
    IntegerType,
    FloatType,
    F16Type,
    F32Type,
    F64Type,
)
from mlir.passmanager import PassManager

from onnx2mlir.dialect import onnx, register_onnx_dialect
from onnx2mlir.passes import register_onnx_to_linag_pass


def get_mlir_type_from_numpy(np_dtype):
    np_array = np.array([], dtype=np_dtype)
    if "int" in str(np_array.dtype):
        elem_bits = np_array.itemsize * 8
        if str(np_array.dtype)[0] == "u":
            elem_type = IntegerType.get_unsigned(elem_bits)
        else:
            elem_dtype = IntegerType.get_signless(elem_bits)
    elif "float16" == str(np_array.dtype):
        elem_type = F16Type()
    elif "float32" == str(np_array.dtype):
        elem_type = F32Type.get()
    elif "float64" == str(np_array.dtype):
        elem_type = F64Type.get()
    else:
        raise TypeError(f"Unsupported numpy datatype: {np_dtype}")
    return elem_dtype


def get_mlir_rankedtensor_from_numpy(np_array):
    elem_type = get_mlir_type_from_numpy(np_array.dtype)
    tensor_type = RankedTensorType.get(
        shape=np_array.shape, element_type=elem_type, loc=Location.unknown()
    )
    return tensor_type, elem_type


def test_onnx_ConstantOp_lower():
    """
    Test ONNX ConstantOp lower.
    """

    EXPECTED_OUTPUT = textwrap.dedent(
        """
        module {
          func.func @main() -> tensor<2x2xi8> {
            %cst = arith.constant dense<[[-128, 8], [-9, 127]]> : tensor<2x2xi8>
            return %cst : tensor<2x2xi8>
          }
        }
        """
    )

    def create_mlir_module():
        with Context() as ctx, Location.unknown() as unk:
            register_onnx_dialect(ctx)
            module = Module.create(unk)
            np_array = np.array([[-128, 8], [-9, 127]], dtype=np.int8)
            tensor_type, elem_type = get_mlir_rankedtensor_from_numpy(np_array)
            func_op = func.FuncOp("main", ([], [tensor_type]))
            with InsertionPoint(func_op.add_entry_block()):
                tensor_attr = DenseElementsAttr.get(
                    np_array,
                    type=tensor_type,
                    context=ctx,
                )
                const = onnx.ConstantOp(tensor_type, value=tensor_attr)
                func.ReturnOp([const.result])
            module.body.append(func_op)
            module.operation.verify()
        return module

    mlir_module = create_mlir_module()

    register_onnx_to_linag_pass()

    with Context() as ctx, Location.unknown() as unk:
        pm = PassManager()
        pm.add("lower-onnx-to-linalg")
        pm.run(mlir_module.operation)
        mlir_module.operation.verify()

    actual_output = str(mlir_module)

    expected_lines = EXPECTED_OUTPUT.strip().splitlines()
    actual_lines = actual_output.strip().splitlines()

    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile="expected.mlir",
        tofile="actual.mlir",
        lineterm="",
    )
    diff_output = "\n".join(list(diff))

    assert not diff_output, f"MLIR output mismatch detected:\n{diff_output}"
