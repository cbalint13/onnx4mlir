###############################################################################
#
#  ONNX2MLIR (ONNX dialect mappings for composable optimizations)
#
#  Authors:
#   Cristian Balint <cristian dot balint at gmail dot com>
#
#  Copyright (c) 2021,2025
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################
# pylint: disable=line-too-long,invalid-name

"""
\file tests/python/conversion/test_onnx_to_linalg_ops.py
\brief Tests for Onnx to Linalg operator lowering
"""

import difflib
import textwrap
import inspect
import pytest
import numpy as np
from mlir.dialects import func
from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
)
from mlir.passmanager import PassManager

from onnx2mlir import support
from onnx2mlir.dialect import onnx, register_onnx_dialect
from onnx2mlir.passes import register_onnx_to_linag_pass


@pytest.mark.parametrize(
    "CALL_OPERATOR",
    [
        getattr(onnx, op)
        for op in dir(onnx)
        if (("ConstantOp" in op) or ("Constant_" in op))
        and inspect.isfunction(getattr(onnx, op))
    ],
)
def test_onnx_ConstantOp_lower(CALL_OPERATOR):
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
            tensor = support.mlir_dense_from_numpy(np_array)
            func_op = func.FuncOp("main", ([], [tensor.type]))
            with InsertionPoint(func_op.add_entry_block()):
                const = CALL_OPERATOR(tensor.type, value=tensor)
                func.ReturnOp([const])
            module.body.append(func_op)
            module.operation.verify()
        return module

    mlir_module = create_mlir_module()

    register_onnx_to_linag_pass()

    with Context(), Location.unknown():
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
