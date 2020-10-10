
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

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

void print(TfLiteIntArray* arr)
{
  for (int i=0; i<arr->size; ++i) {
    printf("%d ", arr->data[i]);
  }
}

// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T, typename T2>
void sort_indexes(const T* v, T2* idx, size_t size)
{
  // initialize original index locations
  std::iota(idx, idx + size, 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx, idx + size,
       [&v](T2 i1, T2 i2) {return v[i1] > v[i2];});
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
  auto input_tensor = interpreter->tensor(inputs[0]);
  auto output_tensor = interpreter->tensor(outputs[0]);
  TfLiteIntArray* input_dim = input_tensor->dims;
  TfLiteIntArray* output_dim = output_tensor->dims;
  printf("input_dim : ");
  print(input_dim);
  printf("\n");
  printf("output_dim : ");
  print(output_dim);
  printf("\n");
  auto input_data = interpreter->typed_input_tensor<uint8_t>(0);
  auto output_data = interpreter->typed_output_tensor<uint8_t>(0);
  int input_width = input_dim->data[1];
  int input_height = input_dim->data[2];
  int input_channels = input_dim->data[3];

  int x,y,n;
  unsigned char *data = stbi_load(imageFilePath, &x, &y, &n, input_channels);
  if (!data) {
    printf("failed to load image : %s\n", imageFilePath);
    return 0;
  }

  stbir_resize_uint8(data, x, y, 0,
                     input_data, input_width, input_height, 0, input_channels);
  stbi_image_free(data);

  status = interpreter->Invoke();

  int output_len = output_dim->data[1];
  std::vector<int> indexes(output_len);
  sort_indexes(output_data, &indexes[0], output_len);

  for (size_t i=0; i<10; ++i) {
    int idx = indexes[i];
    uint8_t score = output_data[idx];
    if (!score)
      break;
    printf("[%zu] : %d, %d\n", i, idx, score);
  }

  return 0;
}

