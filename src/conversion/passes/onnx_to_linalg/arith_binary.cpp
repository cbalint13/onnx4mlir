/******************************************************************+************
 *
 * ONNX2MLIR (ONNX dialect mappings for composable optimizations)
 *
 * Authors:
 *     Cristian Balint <cristian dot balint at gmail dot com>
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
 * \file src/conversion/passes/onnx_to_linalg/arith_binary.cpp
 * \brief ONNX Arith binary operations to Linalg lowering
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Transform/IR/TransformOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "onnx2mlir/common/onnx.hpp"
#include "onnx2mlir/conversion/onnx_passes.hpp"

namespace onnx2mlir::dialect {

mlir::LogicalResult
OnnxToLinalg_ArithBinaryOps(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) {
  auto opName = op->getName().getStringRef();

  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);
  mlir::Value res = op->getResult(0);

  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  auto resType = mlir::dyn_cast<mlir::RankedTensorType>(res.getType());

  if ((!lhsType) || (!rhsType)) {
    return rewriter.notifyMatchFailure(
        op, opName + " operands must be ranked tensor type");
  }

  if (lhsType.getElementType() != rhsType.getElementType()) {
    return rewriter.notifyMatchFailure(
        op, opName + " operands element type are different");
  }

  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, opName + " result must be a ranked tensor type");
  }

  if (opNameBeginsWith(opName, "Pow") &&
      mlir::isa<mlir::IntegerType>(resType.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, opName + " not supported with integer types");
  }

  // Infer broadcasted shape output
  auto outBrdType = getBroadcastShape(lhsType, rhsType);

  if (!outBrdType) {
    return rewriter.notifyMatchFailure(
        op, opName + " operands are not broadcastable");
  }

  if ((outBrdType) && (resType != outBrdType)) {
    return rewriter.notifyMatchFailure(
        op, opName + " result not match operands broadcast");
  }

  // Create an empty tensor for the output
  mlir::Value outBuff = rewriter.create<mlir::tensor::EmptyOp>(
      op->getLoc(), resType.getShape(), resType.getElementType());

  // Create indexing maps for the generic op
  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps;
  mlir::AffineMap lhsMap, rhsMap, resMap;

  // Create identity map for the result
  resMap = rewriter.getMultiDimIdentityMap(resType.getRank());

  // Create the map for the LHS
  llvm::SmallVector<mlir::AffineExpr, 4> lhsExprs;
  mlir::Builder builder(op->getContext());
  mlir::AffineExpr zero = builder.getAffineConstantExpr(0);
  for (unsigned i = 0; i < resType.getRank(); ++i) {
    int64_t lhsDimIndex = lhsType.getRank() - (resType.getRank() - i);
    if (lhsType.getRank() < resType.getRank() - i ||
        lhsType.getDimSize(lhsDimIndex) == 1) {
      lhsExprs.push_back(zero);
    } else {
      lhsExprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  lhsMap = mlir::AffineMap::get(resType.getRank(), 0, lhsExprs,
                                builder.getContext());

  // Create the map for the RHS
  llvm::SmallVector<mlir::AffineExpr, 4> rhsExprs;
  for (unsigned i = 0; i < resType.getRank(); ++i) {
    int64_t rhsDimIndex = rhsType.getRank() - (resType.getRank() - i);
    if (rhsType.getRank() < resType.getRank() - i ||
        rhsType.getDimSize(rhsDimIndex) == 1) {
      rhsExprs.push_back(zero);
    } else {
      rhsExprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  rhsMap = mlir::AffineMap::get(resType.getRank(), 0, rhsExprs,
                                builder.getContext());

  indexingMaps.push_back(lhsMap);
  indexingMaps.push_back(rhsMap);
  indexingMaps.push_back(resMap);

  // Create the arith op with linalg.generic
  auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      op->getLoc(), resType, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{outBuff}, indexingMaps,
      llvm::SmallVector<mlir::utils::IteratorType>(
          resType.getRank(), mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &nest, mlir::Location loc, mlir::ValueRange vals) {
        mlir::Value outOp;
        if (mlir::isa<mlir::FloatType>(resType.getElementType())) {
          if (opNameBeginsWith(opName, "Add"))
            outOp = nest.create<mlir::arith::AddFOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Sub"))
            outOp = nest.create<mlir::arith::SubFOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Mul"))
            outOp = nest.create<mlir::arith::MulFOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Div"))
            outOp = nest.create<mlir::arith::DivFOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Pow"))
            outOp = nest.create<mlir::math::PowFOp>(loc, vals[0], vals[1]);
        } else {
          if (opNameBeginsWith(opName, "Add"))
            outOp = nest.create<mlir::arith::AddIOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Sub"))
            outOp = nest.create<mlir::arith::SubIOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Mul"))
            outOp = nest.create<mlir::arith::MulIOp>(loc, vals[0], vals[1]);
          if (opNameBeginsWith(opName, "Div")) {
            if (resType.getElementType().isUnsignedInteger())
              outOp = nest.create<mlir::arith::DivUIOp>(loc, vals[0], vals[1]);
            if (resType.getElementType().isSignedInteger())
              outOp = nest.create<mlir::arith::DivSIOp>(loc, vals[0], vals[1]);
          }
        }
        nest.create<mlir::linalg::YieldOp>(loc, outOp);
      });

  // Tag for transform optimization
  genericOp->setAttr("transform.target_tag", rewriter.getStringAttr(opName));

  rewriter.replaceOp(op, genericOp);

  return mlir::success();
}

} // namespace onnx2mlir::dialect
