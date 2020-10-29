#define _CRT_SECURE_NO_WARNINGS

#undef NDEBUG

#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/kernels/register_ref.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/kernels/padding.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/builtin_op_data.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cnn.h"

void print(const TfLiteIntArray* arr)
{
  for (int i=0; i<arr->size; ++i) {
    printf("%d ", arr->data[i]);
  }
}

int calc_num_elements(const TfLiteIntArray* dims)
{
  int num = 1;
  for (int i=0; i<dims->size; ++i) {
    num *= dims->data[i];
  }
  return num;
}

int get_type_bytes(TfLiteType type)
{
  switch (type) {
  case kTfLiteNoType: return -1;
  case kTfLiteFloat32: return 4;
  case kTfLiteInt32: return 4;
  case kTfLiteUInt8: return 1;
  case kTfLiteInt64: return 8;
  case kTfLiteString: return -1;
  case kTfLiteBool: return -1;
  case kTfLiteInt16: return 2;
  case kTfLiteComplex64: return 8;
  case kTfLiteInt8: return 1;
  case kTfLiteFloat16: return 2;
  case kTfLiteFloat64: return 8;
  case kTfLiteComplex128: return 16;
  }
  return -1;
}

void filename_append(char* buff, const char* path, const char* trail)
{
  char drive[_MAX_DRIVE];
  char dir[_MAX_DIR];
  char file[_MAX_FNAME];
  char ext[_MAX_EXT];
  _splitpath(path, drive, dir, file, ext);
  strcat(file, trail);
  _makepath(buff, drive, dir, file, ext);
}

void write_to_file(const char* filepath, const TfLiteFloatArray* arr)
{
  FILE* f = fopen(filepath, "wb");
  int32_t sz = arr->size;
  fwrite(&sz, sizeof(sz), 1, f);
  fwrite(arr->data, sizeof(arr->data[0]), sz, f);
  fclose(f);
}

void write_to_file(const char* filepath, const TfLiteIntArray* arr)
{
  FILE* f = fopen(filepath, "wb");
  int32_t sz = arr->size;
  fwrite(&sz, sizeof(sz), 1, f);
  fwrite(arr->data, sizeof(arr->data[0]), sz, f);
  fclose(f);
}

void write_to_file(const char* filepath, const TfLiteTensor* tensor)
{
  FILE* f = fopen(filepath, "wb");

  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    auto affine_quantization = (const TfLiteAffineQuantization*)tensor->quantization.params;
    char buff[256];
    filename_append(buff, filepath, "_quant_scale");
    write_to_file(buff, affine_quantization->scale);
    filename_append(buff, filepath, "_quant_zero_point");
    write_to_file(buff, affine_quantization->zero_point);
  }

  const TfLiteIntArray* dims = tensor->dims;
  int num_elements = calc_num_elements(dims);
  int type_bytes = get_type_bytes(tensor->type);
  fwrite(tensor->data.data, type_bytes, num_elements, f);
  fclose(f);
}

template <typename T>
void write_to_file(const char* filepath, const T* data, size_t len)
{
  FILE* f = fopen(filepath, "wb");
  fwrite(data, sizeof(T), len, f);
  fclose(f);
}

int calc_padding_same_size(int in_size, int stride)
{
  return (int)ceilf((float)in_size / (float)stride);
}

Shape toShape(const TfLiteIntArray* arr)
{
  assert(arr->size == 4);
  Shape ret;
  ret.number = arr->data[0];
  ret.height = arr->data[1];
  ret.width = arr->data[2];
  ret.channel = arr->data[3];
  return ret;
}

void quantize_filter_scale(
  float input_scale,
  float output_scale,
  const TfLiteFloatArray* filter_scale,
  std::vector<int32_t>& output_multiplier,
  std::vector<int32_t>& output_shift
  )
{
  const size_t sz = filter_scale->size;
  const float* data = filter_scale->data;
  output_multiplier.resize(sz);
  output_shift.resize(sz);
  double io_scale = (double)input_scale / (double)output_scale;
  for (size_t i=0; i<sz; ++i) {
    double f = data[i];
    f *= io_scale;
    int shift;
    double f2 = std::frexp(f, &shift);
    int m0 = ((1 << 31) - 1) * f2;
    int n = -shift;
    output_multiplier[i] = m0;
    output_shift[i] = n;
  }
}

void emulate_node_Conv2D(
  tflite::Interpreter* interpreter,
  size_t node_idx,
  const TfLiteNode& node,
  const TfLiteRegistration& node_reg)
{
  const TfLiteConvParams* params = (const TfLiteConvParams*)node.builtin_data;
  const TfLiteIntArray* node_inputs = node.inputs;
  const TfLiteIntArray* node_outputs = node.outputs;
  assert(node_inputs->size == 3);
  assert(node_outputs->size == 1);
  TfLiteTensor* input_tensor = interpreter->tensor(node_inputs->data[0]);
  const TfLiteTensor* filter_tensor = interpreter->tensor(node_inputs->data[1]);
  const TfLiteTensor* bias_tensor = interpreter->tensor(node_inputs->data[2]);
  const TfLiteTensor* output_tensor = interpreter->tensor(node_outputs->data[0]);
  Shape input_shape = toShape(input_tensor->dims);
  Shape filter_shape = toShape(filter_tensor->dims);
  assert(bias_tensor->dims->size == 1);
  int bias_num_elements = bias_tensor->dims->data[0];
  Shape output_shape = toShape(output_tensor->dims);
  assert(bias_num_elements == output_shape.channel);
  assert(input_shape.channel == filter_shape.channel);
  assert(filter_shape.number == output_shape.channel);

  int stride_width = params->stride_width;
  int stride_height = params->stride_height;
  int padding_width_offset;
  int padding_height_offset;
  int padding_width = tflite::ComputePaddingWithOffset(params->stride_width, params->dilation_width_factor, input_shape.width, filter_shape.width, output_shape.width, &padding_width_offset);
  int padding_height = tflite::ComputePaddingWithOffset(params->stride_height, params->dilation_height_factor, input_shape.height, filter_shape.height, output_shape.height, &padding_height_offset);

  const TfLiteAffineQuantization* input_quantization_params = (const TfLiteAffineQuantization*)(input_tensor->quantization.params);
  const TfLiteAffineQuantization* output_quantization_params = (const TfLiteAffineQuantization*)(output_tensor->quantization.params);
  assert(input_quantization_params->scale->size == 1);
  assert(output_quantization_params->scale->size == 1);
  const TfLiteAffineQuantization* filter_quantization_params = (const TfLiteAffineQuantization*)(filter_tensor->quantization.params);
  assert(filter_quantization_params->scale->size == output_shape.channel);
  assert(filter_quantization_params->zero_point->size == output_shape.channel);

  float input_scale = input_quantization_params->scale->data[0];
  int input_zero_point = input_quantization_params->zero_point->data[0];
  float output_scale = output_quantization_params->scale->data[0];
  int output_zero_point = output_quantization_params->zero_point->data[0];
  
  int input_offset = -input_zero_point;
  int output_offset = output_zero_point;
  int activation_min = -128;
  int activation_max = 127;
  std::vector<int32_t> output_multiplier;
  std::vector<int32_t> output_shift;
  quantize_filter_scale(
    input_scale, output_scale,
    filter_quantization_params->scale,
    output_multiplier, output_shift);

  const int8_t* input_data = tflite::GetTensorData<int8_t>(input_tensor);
  const int8_t* filter_data = tflite::GetTensorData<int8_t>(filter_tensor);
  const int32_t* bias_data = tflite::GetTensorData<int32_t>(bias_tensor);
  std::vector<int8_t> output(output_shape.num_elements());
  Conv2D_int8_int8(
    input_shape, input_data,
    filter_shape, filter_data,
    bias_data,
    output_shape, &output[0],
    stride_height, stride_width,
    padding_height, padding_width,
    input_offset, output_offset,
    &output_multiplier[0], &output_shift[0],
    activation_min, activation_max);

  char filename[64];
  sprintf(filename, "node%d_output_ref.dat", node_idx);
  write_to_file(filename, output_tensor);
  sprintf(filename, "node%d_output_emu.dat", node_idx);
  write_to_file(filename, &output[0], output_shape.num_elements());
}

void emulate_node_DepthwiseConv2d(
  tflite::Interpreter* interpreter,
  size_t node_idx,
  const TfLiteNode& node,
  const TfLiteRegistration& node_reg)
{
  const TfLiteDepthwiseConvParams* params = (const TfLiteDepthwiseConvParams*)node.builtin_data;
  const TfLiteIntArray* node_inputs = node.inputs;
  const TfLiteIntArray* node_outputs = node.outputs;
  assert(node_inputs->size == 3);
  assert(node_outputs->size == 1);
  TfLiteTensor* input_tensor = interpreter->tensor(node_inputs->data[0]);
  const TfLiteTensor* weights_tensor = interpreter->tensor(node_inputs->data[1]);
  const TfLiteTensor* bias_tensor = interpreter->tensor(node_inputs->data[2]);
  const TfLiteTensor* output_tensor = interpreter->tensor(node_outputs->data[0]);

  Shape input_shape = toShape(input_tensor->dims);
  Shape weights_shape = toShape(weights_tensor->dims);
  assert(bias_tensor->dims->size == 1);
  int bias_num_elements = bias_tensor->dims->data[0];
  Shape output_shape = toShape(output_tensor->dims);
  assert(bias_num_elements == output_shape.channel);
  assert(input_shape.channel == weights_shape.channel);
  assert(weights_shape.number == 1);

  int stride_width = params->stride_width;
  int stride_height = params->stride_height;
  int padding_width_offset;
  int padding_height_offset;
  int padding_width = tflite::ComputePaddingWithOffset(params->stride_width, params->dilation_width_factor, input_shape.width, weights_shape.width, output_shape.width, &padding_width_offset);
  int padding_height = tflite::ComputePaddingWithOffset(params->stride_height, params->dilation_height_factor, input_shape.height, weights_shape.height, output_shape.height, &padding_height_offset);

  const TfLiteAffineQuantization* input_quantization_params = (const TfLiteAffineQuantization*)(input_tensor->quantization.params);
  const TfLiteAffineQuantization* output_quantization_params = (const TfLiteAffineQuantization*)(output_tensor->quantization.params);
  assert(input_quantization_params->scale->size == 1);
  assert(output_quantization_params->scale->size == 1);
  const TfLiteAffineQuantization* weights_quantization_params = (const TfLiteAffineQuantization*)(weights_tensor->quantization.params);
  assert(weights_quantization_params->scale->size == output_shape.channel);
  assert(weights_quantization_params->zero_point->size == output_shape.channel);

  float input_scale = input_quantization_params->scale->data[0];
  int input_zero_point = input_quantization_params->zero_point->data[0];
  float output_scale = output_quantization_params->scale->data[0];
  int output_zero_point = output_quantization_params->zero_point->data[0];
  
  int input_offset = -input_zero_point;
  int output_offset = output_zero_point;
  int activation_min = -128;
  int activation_max = 127;
  std::vector<int32_t> output_multiplier;
  std::vector<int32_t> output_shift;
  quantize_filter_scale(
    input_scale, output_scale,
    weights_quantization_params->scale,
    output_multiplier, output_shift);

  const int8_t* input_data = tflite::GetTensorData<int8_t>(input_tensor);
  const int8_t* weights_data = tflite::GetTensorData<int8_t>(weights_tensor);
  const int32_t* bias_data = tflite::GetTensorData<int32_t>(bias_tensor);
  std::vector<int8_t> output(output_shape.num_elements());

  DepthwiseConv2D_int8_int8(
    input_shape, input_data,
    weights_shape, weights_data,
    bias_data,
    output_shape, &output[0],
    stride_height, stride_width,
    padding_height, padding_width,
    input_offset, output_offset,
    &output_multiplier[0], &output_shift[0],
    activation_min, activation_max);

  char filename[64];
  sprintf(filename, "node%d_output_ref.dat", node_idx);
  write_to_file(filename, output_tensor);
  sprintf(filename, "node%d_output_emu.dat", node_idx);
  write_to_file(filename, &output[0], output_shape.num_elements());
}

void emulate_node(tflite::Interpreter* interpreter, size_t node_idx)
{
  const tflite::Subgraph& graph = interpreter->primary_subgraph();
  size_t num_nodes = graph.nodes_size();
  assert(node_idx < num_nodes);
  const auto& nodes = graph.nodes_and_registration();
  const auto& pair = nodes[node_idx];
  const TfLiteNode& node = pair.first;
  const TfLiteRegistration& node_reg = pair.second;

  switch (node_reg.builtin_code) {
  //case kTfLiteBuiltinAdd:
  //  break;
  //case kTfLiteBuiltinAveragePool2d:
  //  break;
  //case kTfLiteBuiltinConcatenation:
  //  break;
  case kTfLiteBuiltinConv2d:
    emulate_node_Conv2D(interpreter, node_idx, node, node_reg);
    break;
  case kTfLiteBuiltinDepthwiseConv2d:
    emulate_node_DepthwiseConv2d(interpreter, node_idx, node, node_reg);
    break;
  default:
    assert(false);
    break;
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

  // efficientnet-lite0-int8.tflite
  
  auto inputs = interpreter->inputs();
  TfLiteTensor* input_tensor = interpreter->tensor(inputs[0]);
  const TfLiteIntArray* input_dim = input_tensor->dims;
  assert(input_tensor->quantization.type == kTfLiteAffineQuantization);
  const TfLiteAffineQuantization* input_quantization_params = (const TfLiteAffineQuantization*)input_tensor->quantization.params;
  float input_scale = input_quantization_params->scale->data[0];
  int input_zero_point = input_quantization_params->zero_point->data[0];
  printf("input_dim : ");
  print(input_dim);
  printf("\n");
  auto graph_input_data = interpreter->typed_input_tensor<uint8_t>(0);
  int input_width = input_dim->data[1];
  int input_height = input_dim->data[2];
  int input_channels = input_dim->data[3];

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

  memcpy(graph_input_data, data, input_width * input_height * input_channels);
  stbi_image_free(data);

  status = interpreter->Invoke();

  emulate_node(interpreter.get(), 1);
  emulate_node(interpreter.get(), 2);

  auto outputs = interpreter->outputs();
  const TfLiteTensor* output_tensor = interpreter->tensor(outputs[0]);
  const TfLiteIntArray* output_dim = output_tensor->dims;
  assert(output_tensor->quantization.type == kTfLiteAffineQuantization);
  const TfLiteAffineQuantization* output_quantization_params = (const TfLiteAffineQuantization*)output_tensor->quantization.params;
  float output_scale = output_quantization_params->scale->data[0];
  int output_zero_point = output_quantization_params->zero_point->data[0];
  printf("output_dim : ");
  print(output_dim);
  printf("\n");
  auto output_data = interpreter->typed_output_tensor<int8_t>(0);
  int output_width = output_dim->data[1];
  int output_height = output_dim->data[2];
  int output_channels = output_dim->data[3];

  return 0;
}

