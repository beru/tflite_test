#pragma once

#include <stdint.h>
#include <assert.h>
#include <algorithm>

enum class Padding {
  same,
  valid,
};

enum class TensorLayout {
  NHWC,
  NCHW,
};

struct Shape
{
  int number;
  int width;
  int height;
  int channel;
  TensorLayout layout;

  Shape(int number = 1, int height = 1, int width = 1, int channel = 1, TensorLayout layout = TensorLayout::NHWC)
    :
    number(number),
    height(height),
    width(width),
    channel(channel),
    layout(layout)
  {
  }

  int num_elements() const {
    return number * width * height * channel;
  }

  int offset(int n, int y, int x, int c) const {
    assert(n < number);
    assert(y < height);
    assert(x < width);
    assert(c < channel);

    if (layout == TensorLayout::NCHW) {
      return n * channel * height * width + c * height * width + y * width + x;
    }else if (layout == TensorLayout::NHWC) {
      return n * height * width * channel + y * width * channel + x * channel + c;
    }else {
      assert(false);
      return 0;
    }
  }
};

template <typename T>
void Conv2D(
  const Shape input_shape, const T* input_values,
  const Shape filter_shape, const T* filter_values,
  const T* bias_values,
  const Shape output_shape, T* output_values,
  const int stride_height, const int stride_width,
  const int padding_height, const int padding_width
  )
{
  const int input_width = input_shape.width;
  const int input_height = input_shape.height;
  const int input_depth = input_shape.channel;
  const int output_height = output_shape.height;
  const int output_width = output_shape.width;
  const int output_depth = output_shape.channel;
  const int filter_width = filter_shape.width;
  const int filter_height = filter_shape.height;

  for (int out_y=0; out_y<output_height; ++out_y) {
    const int in_y_start = out_y * stride_height - padding_height;
    for (int out_x=0; out_x<output_width; ++out_x) {
      const int in_x_start = out_x * stride_width - padding_width;
      for (int out_ch=0; out_ch<output_depth; ++out_ch) {
        T sum = 0;
        for (int filter_y=0; filter_y<filter_height; ++filter_y) {
          const int in_y = in_y_start + filter_y;
          if (in_y < 0 || in_y >= input_height)
            continue;
          for (int filter_x=0; filter_x<filter_width; ++filter_x) {
            const int in_x = in_x_start + filter_x;
            if (in_x < 0 || in_x >= input_width)
              continue;
            for (int in_ch=0; in_ch<input_depth; ++in_ch) {
              T input_value = input_values[input_shape.offset(0, in_y, in_x, in_ch)];
              T filter_value = filter_values[filter_shape.offset(out_ch, filter_y, filter_x, in_ch)];
              sum += filter_value * input_value;
            }
          }
        }
        T bias = bias_values[out_ch];
        sum += bias;
        output_values[output_shape.offset(0, out_y, out_x, out_ch)] = sum;
      }
    }
  }
}

inline
void Conv2D_int8_int8(
  const Shape input_shape, const int8_t* input_values,
  const Shape filter_shape, const int8_t* filter_values,
  const int32_t* bias_values,
  const Shape output_shape, int8_t* output_values,
  const int stride_height, const int stride_width,
  const int padding_height, const int padding_width,
  const int32_t input_offset, const int32_t output_offset,
  const int32_t* output_multiplier, const int32_t* output_shift,
  const int32_t activation_min, const int32_t activation_max
  )
{
  assert(input_shape.num_elements() > 0);
  assert(filter_shape.num_elements() > 0);
  assert(output_shape.num_elements() > 0);
  assert(input_values != nullptr);
  assert(filter_values != nullptr);
  assert(output_values != nullptr);
  assert(stride_height >= 1);
  assert(stride_width >= 1);
  assert(padding_height >= 0);
  assert(padding_width >= 0);
  assert(output_multiplier != nullptr);
  assert(output_shift != nullptr);

  const int input_width = input_shape.width;
  const int input_height = input_shape.height;
  const int input_depth = input_shape.channel;
  const int output_height = output_shape.height;
  const int output_width = output_shape.width;
  const int output_depth = output_shape.channel;
  const int filter_width = filter_shape.width;
  const int filter_height = filter_shape.height;

  for (int out_y=0; out_y<output_height; ++out_y) {
    const int in_y_start = out_y * stride_height - padding_height;
    for (int out_x=0; out_x<output_width; ++out_x) {
      const int in_x_start = out_x * stride_width - padding_width;
      for (int out_ch=0; out_ch<output_depth; ++out_ch) {
        int32_t sum = 0;
        const int32_t m0 = output_multiplier[out_ch];
        const int32_t n = output_shift[out_ch];
        for (int filter_y=0; filter_y<filter_height; ++filter_y) {
          const int in_y = in_y_start + filter_y;
          if (in_y < 0 || in_y >= input_height)
            continue;
          for (int filter_x=0; filter_x<filter_width; ++filter_x) {
            const int in_x = in_x_start + filter_x;
            if (in_x < 0 || in_x >= input_width)
              continue;
            for (int in_ch=0; in_ch<input_depth; ++in_ch) {
              int32_t input_value = input_values[input_shape.offset(0, in_y, in_x, in_ch)];
              int32_t filter_value = filter_values[filter_shape.offset(out_ch, filter_y, filter_x, in_ch)];
              sum += filter_value * (input_value + input_offset);
            }
          }
        }
        sum += bias_values[out_ch];
        int64_t half = 1LL << (30 + n);
        sum = (int32_t)(((int64_t)sum * m0 + half) >> (31 + n));
        sum += output_offset;
        sum = std::max(sum, activation_min);
        sum = std::min(sum, activation_max);
        output_values[output_shape.offset(0, out_y, out_x, out_ch)] = (int8_t)sum;
      }
    }
  }
}

inline
void Conv2D_uint8_uint8(
  const Shape input_shape, const uint8_t* input_values,
  const Shape filter_shape, const int8_t* filter_values,
  const int32_t* bias_values,
  const Shape output_shape, uint8_t* output_values,
  const int stride_height, const int stride_width,
  const int padding_height, const int padding_width,
  const int32_t* output_multiplier, const int32_t* output_shift,
  const int32_t activation_min, const int32_t activation_max
  )
{
  const int input_width = input_shape.width;
  const int input_height = input_shape.height;
  const int input_depth = input_shape.channel;
  const int output_height = input_shape.height;
  const int output_width = output_shape.width;
  const int output_depth = output_shape.channel;
  const int filter_width = filter_shape.width;
  const int filter_height = filter_shape.height;

  for (int out_y=0; out_y<output_height; ++out_y) {
    const int in_y_start = out_y * stride_height - padding_height;
    for (int out_x=0; out_x<output_width; ++out_x) {
      const int in_x_start = out_x * stride_width - padding_width;
      for (int out_ch=0; out_ch<output_depth; ++out_ch) {
        int32_t sum = 0;
        for (int filter_y=0; filter_y<filter_height; ++filter_y) {
          const int in_y = in_y_start + filter_y;
          if (in_y < 0 || in_y >= input_height)
            continue;
          for (int filter_x=0; filter_x<filter_width; ++filter_x) {
            const int in_x = in_x_start + filter_x;
            if (in_x < 0 || in_x >= input_width)
              continue;
            for (int in_ch=0; in_ch<input_depth; ++in_ch) {
              int32_t input_value = input_values[input_shape.offset(0, in_y, in_x, in_ch)];
              int32_t filter_value = filter_values[filter_shape.offset(out_ch, filter_y, filter_x, in_ch)];
              sum += filter_value * input_value;
            }
          }
        }
        sum += bias_values[out_ch];
        const int32_t m0 = output_multiplier[out_ch];
        const int32_t n = output_shift[out_ch];
        sum = (int32_t)(((int64_t)sum * m0) >> n);
        sum = std::max(sum, activation_min);
        sum = std::min(sum, activation_max);
        output_values[output_shape.offset(0, out_y, out_x, out_ch)] = (uint8_t)sum;
      }
    }
  }
}

