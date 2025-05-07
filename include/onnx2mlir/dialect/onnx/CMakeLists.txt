add_mlir_dialect(Onnx onnx)
add_mlir_doc(OnnxDialect OnnxDialect onnx/ -gen-dialect-doc)
add_mlir_doc(OnnxOps OnnxOps onnx/ -gen-op-doc)
add_mlir_doc(OnnxTypes OnnxTypes onnx/ -gen-type-doc)

#set(LLVM_TARGET_DEFINITIONS Passes.td)
#mlir_tablegen(Passes.h.inc -gen-pass-decls -name RelayOpt)
#add_public_tablegen_target(mlir_relay_pass_inc_gen)
#add_mlir_doc(Passes RelayPasses ./ -gen-pass-doc)
