#!/usr/bin/env python3

import re
import sys
import onnx
from collections import defaultdict, OrderedDict
from datetime import datetime
from onnx import defs


def get_mlir_types_from_str(type_strs, schema_constraints):
  onnx_to_mlir_types = {
    "(": "<[",
    ")": "]>",
    "tensor": "TensorOf",
    "seq": "SeqOf",
    "map": "TupleOf",
    "bool": "I1",
    "uint4": "UI<4>",
    "uint8": "UI8",
    "uint16": "UI16",
    "uint32": "UI32",
    "uint64": "UI64",
    "int4": "I<4>",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "double": "F64",
    "float": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "float8e4m3fn": "F8E4M3FN",
    "float8e4m3fnuz": "F8E4M3FNUZ",
    "float8e5m2": "F8E5M2",
    "float8e5m2fnuz": "F8E5M2FNUZ",
    "float4e2m1": "F4E2M1FN",
    "complex64": "Complex<F32>",
    "complex128": "Complex<F64>",
    "string": "StringType"
  }

  attr_constraints = []

  # pick up direct constraint
  for idx, param in enumerate(type_strs):
    attr_constraints.append(param.type_str)
    # pick up matched constraints
    for attr in schema_constraints:
      if (attr.type_param_str == param.type_str):
        attr_constraints = attr.allowed_type_strs

  # build mlir attributes
  mlir_types = "AnyTypeOf<[" if (len(attr_constraints) > 1) else ""
  for idx, attr in enumerate(attr_constraints):

    if attr.find("optional") == 0:
      attr = attr.replace("optional(", "OptOf<", 1)
      attr = attr[:-1] + ">"

    pattern = '|'.join(re.escape(key) for key in sorted(onnx_to_mlir_types.keys(), key=len, reverse=True))
    mlir_type_str = re.sub(pattern, lambda m: onnx_to_mlir_types[m.group(0)], attr)
    mlir_types += f"{mlir_type_str}" + (", " if idx+1 < len(attr_constraints) else "")
  mlir_types += "]>" if (len(attr_constraints) > 1) else ""

  return mlir_types


def main():

#  for domain, support_map in build_operator_schemas():
#    print(domain)
#  exit(0)

  inc = open("OnnxOps.td.inc", "w")
  inc.write("/********************************************************\n")
  inc.write(" *   ONNX version [%s]%s*\n" % (onnx.__version__, " " * 30))
  inc.write(" *   Generated at [%s]%s*\n" % (datetime.now(), " " * 10))
  inc.write(" ********************************************************/\n")

  ops_versions = {}
  # map Ops versions
  for schema in defs.get_all_schemas_with_history():
    if schema.name not in ops_versions:
      ops_versions[schema.name] = []
    ops_versions[schema.name].append(int(schema.since_version))

  # iterate Ops
  for schema in defs.get_all_schemas_with_history():

   #if schema.name == "NonMaxSuppression":
   if (schema.name == "Add") or \
      (schema.name == "Constant"):

    # use latest version Op
    if schema.since_version != max(ops_versions[schema.name]):
      continue

    inc.write(f'\n')
    inc.write(f'/// {schema.name}\n')
    inc.write(f'def Onnx_{schema.name}Op : Onnx_Op<"{schema.name}", []> {{\n')
    inc.write(f'  let summary = "ONNX {schema.name} operation";\n')
    inc.write(f'  let description = [{{\n')
    doctxt = "\n".join([("  " + line.lstrip()) if line else "" for line in schema.doc.splitlines() if line])
    doctxt = doctxt.split('.')[0]
    if '.' not in doctxt[-3:] != '.': doctxt += '.'
    inc.write(f'    {doctxt}\n')
    inc.write(f'  }}];\n')
    inc.write(f'\n')
    # arguments
    mlir_types = get_mlir_types_from_str(schema.inputs, schema.type_constraints)
    inc.write(f'  let arguments = (ins ')
    if len(schema.inputs) == 0:
      inc.write(f'OptionalAttr<AnyAttr>: $value')
    else:
      for idx, inp in enumerate(schema.inputs):
        inc.write(f'{mlir_types}: ${inp.name}%s'
          % ('' if idx+1 == len(schema.inputs) else ', '))
    inc.write(');\n')
    # results
    mlir_types = get_mlir_types_from_str(schema.outputs, schema.type_constraints)
    inc.write(f'  let results = (outs ')
    for idx, out in enumerate(schema.outputs):
      inc.write(f'{mlir_types}: ${out.name}%s'
        % ('' if idx+1 == len(schema.outputs) else ', '))
    inc.write(');\n')
    inc.write(f'}}\n')


# /// Constant
# def Onnx_ConstantOp : Onnx_Op<"Constant", [ConstantLike]> {
#   let summary = "ONNX Constant operation";
#   let description = [{
#     This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
#     or value_* must be specified.
#   }];
# 
#   let arguments = (ins ElementsAttr : $value);
#   let results = (outs AnyStaticShapeTensor : $result);
# }

# DBG [][SupportType.COMMON][Constant]
#    ATTR [{'value': OpSchema.Attribute(
#                      name='value', type=<AttrType.TENSOR: 4>,
#                      description='The value for the elements of the output tensor.',
#                      default_value=,
#                      required=True)}]
#    DOC [A constant tensor.]
#    OUT#0 [OpSchema.FormalParameter(
#                      name='output',
#                      type_str='T',
#                      description='Output tensor containing the same value of the provided tensor.',
#                      param_option=<FormalParameterOption.Single: 0>,
#                      is_homogeneous=True,
#                      min_arity=1,
#                      differentiation_category=<DifferentiationCategory.Unknown: 0>)]


    print("\n====================[%s]=======================\n" % schema.name)
#    print(dir(schema.outputs))
    print("DBG [%s][%s][%s] [%s]" % (schema.domain, schema.support_level, schema.name, schema.since_version))
    print("   ATTR [%s]" % (schema.attributes))
#    print("   DOC [%s]" % (schema.doc))
    for idx, inp in enumerate(schema.inputs):
      print("   IN#%i [%s]" % (idx, inp))
    for idx, out in enumerate(schema.outputs):
      print("   OUT#%i [%s]" % (idx, out))
    print("   FUNCBODY [%s]" % (schema.function_body))
    print("   TYPECONSTRAINTS [%s]" % (schema.type_constraints))

  inc.close()

if __name__ == "__main__":
  main()
