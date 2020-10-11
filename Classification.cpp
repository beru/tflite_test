#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/kernels/register_ref.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void print(TfLiteIntArray* arr)
{
  for (int i=0; i<arr->size; ++i) {
    printf("%d ", arr->data[i]);
  }
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    printf("usage : model_file image_file\n");
    return 0;
  }

  const char* modelFilePath = argv[1];
  const char* imageFilePath = argv[2];

  auto model = tflite::FlatBufferModel::BuildFromFile(modelFilePath);

  // Build the interpreter
  //tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::builtin::BuiltinRefOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

//  tflite::PrintInterpreterState(interpreter.get());

  TfLiteStatus status = interpreter->AllocateTensors();
  auto inputs = interpreter->inputs();
  auto outputs = interpreter->outputs();

  // efficientnet-lite0-int8.tflite
  const tflite::Subgraph& graph = interpreter->primary_subgraph();
  
  TfLiteTensor* input_tensor = interpreter->tensor(inputs[0]);
  TfLiteTensor* output_tensor = interpreter->tensor(outputs[0]);
  TfLiteIntArray* input_dim = input_tensor->dims;
  TfLiteIntArray* output_dim = output_tensor->dims;

  const TfLiteQuantization& input_quantization = input_tensor->quantization;
  const TfLiteQuantization& output_quantization = output_tensor->quantization;
  assert(input_quantization.type == kTfLiteAffineQuantization);
  assert(output_quantization.type == kTfLiteAffineQuantization);
  const TfLiteAffineQuantization* input_affine_quantization = (const TfLiteAffineQuantization*)input_quantization.params;
  const TfLiteAffineQuantization* output_affine_quantization = (const TfLiteAffineQuantization*)output_quantization.params;
  float input_scale = input_affine_quantization->scale->data[0];
  int input_zero_point = input_affine_quantization->zero_point->data[0];
  float output_scale = output_affine_quantization->scale->data[0];
  int output_zero_point = output_affine_quantization->zero_point->data[0];

  printf("input_dim : ");
  print(input_dim);
  printf("\n");
  printf("output_dim : ");
  print(output_dim);
  printf("\n");
  auto input_data = interpreter->typed_input_tensor<uint8_t>(0);
  auto output_data = interpreter->typed_output_tensor<int8_t>(0);
  int input_width = input_dim->data[1];
  int input_height = input_dim->data[2];
  int input_channels = input_dim->data[3];
  int output_width = output_dim->data[1];
  int output_height = output_dim->data[2];
  int output_channels = output_dim->data[3];

  size_t num_nodes = graph.nodes_size();
  const auto& nodes = graph.nodes_and_registration();

  int node0_output_idx = nodes[0].first.outputs->data[0];
  //int node1_output_idx = nodes[1].first.outputs->data[0];
  TfLiteTensor* node0_output_tensor = interpreter->tensor(node0_output_idx);
  output_affine_quantization = (const TfLiteAffineQuantization*)(node0_output_tensor->quantization.params);

  int x,y,n;
  unsigned char *data = stbi_load(imageFilePath, &x, &y, &n, input_channels);
  if (!data) {
    printf("failed to load image : %s\n", imageFilePath);
    return 0;
  }

  if (x != input_width || y != input_height || n != input_channels) {
    stbi_image_free(data);
    printf("input_width must be %d\n", input_width);
    printf("input_height must be %d\n", input_height);
    printf("input_channels must be %d\n", input_channels);
    return 0;
  }

  memcpy(input_data, data, input_width * input_height * input_channels);
  stbi_image_free(data);

  status = interpreter->Invoke();

  FILE* f = fopen("output.dat", "wb");
  fwrite(output_data, 1, output_width * output_height * output_channels, f);
  fclose(f);

  return 0;
}

