add_mlir_python_extension(onnx2mlir_py_onnx_passes _onnx2mlirPassesOnnx
  SOURCES
    src/conversion/passes/onnx_pass_bindings.cpp
  OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/passes
  LINK_LIBS
    onnx2mlir_onnx_linalg_pass
    MLIRPythonCAPI
)

add_mlir_python_extension(onnx2mlir_py_onnx_dialect _onnx2mlirDialectsOnnx
  SOURCES
    src/dialect/onnx_dialect_bindings.cpp
  OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/dialect
  LINK_LIBS
    onnx2mlir_onnx_dialect
    MLIRPythonCAPI
)
