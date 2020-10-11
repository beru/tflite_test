#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void offset_elements(uint8_t* input_data, int8_t* output_data, int num_elements, int offset)
{
  for (int i=0; i<num_elements; ++i) {
    output_data[i] = input_data[i] - 128;
  }
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
  std::vector<int8_t> output(num_elements);

  int8_t* output_data = &output[0];

  offset_elements(data, output_data, num_elements, -128);

  stbi_image_free(data);

  FILE* f = fopen("output.dat", "wb");
  fwrite(output_data, 1, num_elements, f);
  fclose(f);

  return 0;
}

