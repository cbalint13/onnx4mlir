/******************************************************************************
 *
 * Copyright (c) 2021,2025
 *
 *     Cristian Balint <cristian dot balint at gmail dot com>
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *****************************************************************************/

/*!
 * \file OnnxOps.hpp
 *
 */

#ifndef ONNX2MLIR_DIALECT_ONNX_ONNXOPS_HPP_
#define ONNX2MLIR_DIALECT_ONNX_ONNXOPS_HPP_

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "onnx2mlir/dialect/onnx/OnnxOps.h.inc"
#endif // ONNX2MLIR_DIALECT_ONNX_ONNXOPS_HPP_
