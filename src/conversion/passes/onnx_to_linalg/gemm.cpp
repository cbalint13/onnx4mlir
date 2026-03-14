/******************************************************************+************
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 * Authors:
 * Cristian Balint <cristian dot balint at gmail dot com>
 *
 * Copyright (c) 2021,2025
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
 * \file src/conversion/passes/onnx_to_linalg/gemm.cpp
 * \brief ONNX Gemm operation to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/dialect/onnx/Onnx.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult OnnxToLinalg_GemmOp(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto opName = op->getName().getStringRef();

  // Parse Operands
  mlir::Value A = op->getOperand(0);
  mlir::Value B = op->getOperand(1);
  mlir::Value C = op->getNumOperands() > 2 ? op->getOperand(2) : mlir::Value();

  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(A.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(B.getType());
  auto resType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());

  if (!aType || !bType || !resType) {
    return mlir::emitError(loc, opName + " operand must be ranked tensor type");
  }

  mlir::Type elementType = resType.getElementType();
  bool isFloat = mlir::isa<mlir::FloatType>(elementType);
  bool isInt = mlir::isa<mlir::IntegerType>(elementType);

  if (!isFloat && !isInt) {
    return mlir::emitError(loc, opName + " supports float or integer tensors");
  }

  // Parse Attributes
  float alpha = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
    alpha = attr.getValueAsDouble();

  float beta = 1.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
    beta = attr.getValueAsDouble();

  int64_t transA = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transA"))
    transA = attr.getInt();

  int64_t transB = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transB"))
    transB = attr.getInt();

  int64_t broadcast = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("broadcast"))
    broadcast = attr.getInt();

  // Helper for Transposition
  auto handleTranspose = [&](mlir::Value input, int64_t trans) -> mlir::Value {
    if (trans == 0)
      return input;
    auto type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = type.getShape();
    llvm::SmallVector<int64_t, 2> newShape{shape[1], shape[0]};

    auto init = mlir::tensor::EmptyOp::create(rewriter, loc, newShape,
                                              type.getElementType());
    llvm::SmallVector<int64_t, 2> perms{1, 0};
    auto permsAttr = rewriter.getDenseI64ArrayAttr(perms);

    auto transposeOp = mlir::linalg::TransposeOp::create(rewriter, loc, input,
                                                         init, permsAttr);
    return transposeOp->getResult(0);
  };

  mlir::Value finalA = handleTranspose(A, transA);
  mlir::Value finalB = handleTranspose(B, transB);

  // Initialize Output Buffer (handles Beta * C)
  auto outTBuff = mlir::tensor::EmptyOp::create(
      rewriter, loc, resType.getShape(), resType.getElementType());

  mlir::Value outBuff;
  bool hasC = C && !mlir::isa<mlir::NoneType>(C.getType());

  if (hasC) {
    auto cType = mlir::dyn_cast<mlir::RankedTensorType>(C.getType());

    mlir::SmallVector<mlir::AffineMap> maps;
    mlir::SmallVector<mlir::AffineExpr> cExprs;

    if (broadcast == 0 && cType.getShape() != resType.getShape()) {
      return mlir::emitError(
          loc, opName + " broadcast=0 requires C shape to match result shape");
    }

    for (unsigned i = 0; i < cType.getRank(); ++i) {
      int64_t resDimIdx = resType.getRank() - cType.getRank() + i;
      cExprs.push_back(rewriter.getAffineDimExpr(resDimIdx));
    }
    maps.push_back(
        mlir::AffineMap::get(resType.getRank(), 0, cExprs, op->getContext()));
    maps.push_back(rewriter.getMultiDimIdentityMap(resType.getRank()));

    mlir::SmallVector<mlir::utils::IteratorType> iterators(
        resType.getRank(), mlir::utils::IteratorType::parallel);

    auto broadcastOp = mlir::linalg::GenericOp::create(
        rewriter, loc, resType, mlir::ValueRange{C}, mlir::ValueRange{outTBuff},
        maps, iterators,
        [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
          mlir::Value val = args[0];
          // Handle Beta scaling
          if (std::abs(beta - 1.0f) > 1e-6) {
            if (isFloat) {
              mlir::Value bConst = mlir::arith::ConstantOp::create(
                  nest, l, nest.getFloatAttr(elementType, beta));
              val = mlir::arith::MulFOp::create(nest, l, val, bConst);
            } else {
              mlir::Value bConst = mlir::arith::ConstantOp::create(
                  nest, l,
                  nest.getIntegerAttr(elementType, static_cast<int64_t>(beta)));
              val = mlir::arith::MulIOp::create(nest, l, val, bConst);
            }
          }
          mlir::linalg::YieldOp::create(nest, l, val);
        });
    outBuff = broadcastOp->getResult(0);
  } else {
    auto zeroAttr = rewriter.getZeroAttr(elementType);
    auto constantZero =
        mlir::arith::ConstantOp::create(rewriter, loc, zeroAttr);
    outBuff = mlir::linalg::FillOp::create(rewriter, loc,
                                           mlir::ValueRange{constantZero},
                                           mlir::ValueRange{outTBuff})
                  ->getResult(0);
  }

  // Lower to Linalg Matmul
  auto matmulOp = mlir::linalg::MatmulOp::create(
      rewriter, loc, mlir::TypeRange{resType}, mlir::ValueRange{finalA, finalB},
      mlir::ValueRange{outBuff});

  mlir::Value result = matmulOp->getResult(0);

  // Apply Alpha Scaling
  if (std::abs(alpha - 1.0f) > 1e-6) {
    auto alphaTBuff = mlir::tensor::EmptyOp::create(
        rewriter, loc, resType.getShape(), elementType);

    mlir::SmallVector<mlir::AffineMap> alphaMaps(
        2, rewriter.getMultiDimIdentityMap(resType.getRank()));
    mlir::SmallVector<mlir::utils::IteratorType> alphaIters(
        resType.getRank(), mlir::utils::IteratorType::parallel);

    auto alphaOp = mlir::linalg::GenericOp::create(
        rewriter, loc, resType, mlir::ValueRange{result},
        mlir::ValueRange{alphaTBuff}, alphaMaps, alphaIters,
        [&](mlir::OpBuilder &nest, mlir::Location l, mlir::ValueRange args) {
          mlir::Value scaled;
          if (isFloat) {
            mlir::Value aConst = mlir::arith::ConstantOp::create(
                nest, l, nest.getFloatAttr(elementType, alpha));
            scaled = mlir::arith::MulFOp::create(nest, l, args[0], aConst);
          } else {
            mlir::Value aConst = mlir::arith::ConstantOp::create(
                nest, l,
                nest.getIntegerAttr(elementType, static_cast<int64_t>(alpha)));
            scaled = mlir::arith::MulIOp::create(nest, l, args[0], aConst);
          }
          mlir::linalg::YieldOp::create(nest, l, scaled);
        });
    result = alphaOp->getResult(0);
  }

  matmulOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, result);
  return mlir::success();
}

} // namespace onnx2mlir::dialect
