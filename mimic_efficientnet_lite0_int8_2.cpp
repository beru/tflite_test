#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <io.h>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cnn.h"

void offset_elements(uint8_t* input_data, int8_t* output_data, int num_elements, int offset)
{
  for (int i=0; i<num_elements; ++i) {
    output_data[i] = input_data[i] - 128;
  }
}

int calc_padding_same_size(int in_size, int stride)
{
  return (int)ceilf((float)in_size / (float)stride);
}

template <typename T>
bool read_elements_from_file(const char* filename, std::vector<T>& buff)
{
  FILE* f = fopen(filename, "rb");
  if (!f) {
    return false;
  }
  long len = _filelength(_fileno(f));
  assert(len % sizeof(T) == 0);
  long num_elements = len / sizeof(T);
  buff.resize(num_elements);
  fread(&buff[0], sizeof(T), num_elements, f);
  fclose(f);
  return true;
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("usage : image_file\n");
    return 0;
  }

  const char* imageFilePath = argv[1];

  int x,y,n;
  unsigned char *data = stbi_load(imageFilePath, &x, &y, &n, 3);
  if (!data) {
    printf("failed to load image : %s\n", imageFilePath);
    return 0;
  }

  const int num_elements = x * y * n;
  std::vector<int8_t> node0_output(num_elements);
  int8_t* node0_output_data = &node0_output[0];
  offset_elements(data, node0_output_data, num_elements, -128);
  stbi_image_free(data);

  std::vector<int8_t> node1_input;
  std::vector<int8_t> node1_filter;
  std::vector<int32_t> node1_bias;
  read_elements_from_file("node1_input0.dat", node1_input);
  read_elements_from_file("node1_input1.dat", node1_filter);
  read_elements_from_file("node1_input2.dat", node1_bias);

  Shape node1_input_shape {1, 224, 224, 3};
  Shape node1_filter_shape {32, 3, 3, 3};
  int node1_bias_num_elements = 32;
  Shape node1_output_shape;
  node1_output_shape.number = 1;
  int node1_stride_width = 2;
  int node1_stride_height = 2;
  int node1_padding_width = 0;
  int node1_padding_height = 0;
  node1_output_shape.width = calc_padding_same_size(x, node1_stride_width);
  node1_output_shape.height = calc_padding_same_size(y, node1_stride_height);
  node1_output_shape.channel = 32;
  int node1_input_offset = 0;
  int node1_output_offset = 0;
  int node1_activation_min = 0;
  int node1_activation_max = 255;
  int output_multiplier[3];
  int output_shift[3];
  std::vector<int8_t> node1_output(node1_output_shape.num_elements());

  Conv2D_int8_int8(
    node1_input_shape, &node1_input[0],
    node1_filter_shape, &node1_filter[0],
    &node1_bias[0],
    node1_output_shape, &node1_output[0],
    node1_stride_height, node1_stride_width,
    node1_padding_height, node1_padding_width,
    node1_input_offset, node1_output_offset,
    output_multiplier, output_shift,
    node1_activation_min, node1_activation_max);

  return 0;
}

