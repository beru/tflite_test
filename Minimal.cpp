
#include <iostream>
#include <cstdio>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/kernels/register_ref.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

int main(int argc, char* argv[])
{
  auto model = tflite::FlatBufferModel::BuildFromFile("mobilenet_v1_1.0_224_quant.tflite");

  // Build the interpreter
  //tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::builtin::BuiltinRefOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

  // Allocate tensor buffers.
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Run inference
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());
  
  return 0;
}

