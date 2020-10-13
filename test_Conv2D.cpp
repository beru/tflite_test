
#include "doctest.h"

#include "cnn.h"

TEST_CASE("Conv2D input=1x1 filter=1x1 output=1x1")
{
  Shape input_shape(1, 1, 1, 1);
  float input_values[] = { 1.0f, };

  Shape filter_shape(1, 1, 1, 1);
  float filter_values[] = { 1.0f, };

  float bias_values[] = { -1.0f, };

  Shape output_shape(1, 1, 1, 1);
  float output_values[1];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    0, 0
  );

  CHECK(output_values[0] == 0.0f);
}

TEST_CASE("Conv2D input=1x1x1 filter=2x1x1 output=1x1x2")
{
  Shape input_shape(1, 1, 1, 1);
  float input_values[] = { 1.0f, };

  Shape filter_shape(2, 1, 1, 1);
  float filter_values[] = { 1.0f, -1.0f, };

  float bias_values[] = { -1.0f, +123.0f, };

  Shape output_shape(1, 1, 1, 2);
  float output_values[2];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    0, 0
  );

  float expected_output_values[2] = {
    0.0f, 122.0f,
  };

  CHECK(std::equal(std::begin(output_values), std::end(output_values), std::begin(expected_output_values)));
}

TEST_CASE("Conv2D input=1x1x2 filter=1x1x2 output=1x1x1")
{
  Shape input_shape(1, 1, 1, 2);
  float input_values[] = { 1.0f, 1.0f, };

  Shape filter_shape(1, 1, 1, 2);
  float filter_values[] = { 1.0f, 1.0f, };

  float bias_values[] = { 123.0f, };

  Shape output_shape(1, 1, 1, 1);
  float output_values[1];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    0, 0
  );

  float expected_output_values[1] = {
    125.0f,
  };

  CHECK(std::equal(std::begin(output_values), std::end(output_values), std::begin(expected_output_values)));
}

TEST_CASE("Conv2D input=3x3 filter=3x3 output=1x1 padding=valid")
{
  Shape input_shape(1, 3, 3, 1);
  float input_values[9];
  std::fill_n(input_values, 9, 1.0f);

  Shape filter_shape(1, 3, 3, 1);
  float filter_values[9];
  std::fill_n(filter_values, 9, 1.0f);

  float bias_values[1] = { 0.0f, };

  Shape output_shape(1, 1, 1, 1);
  float output_values[1];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    0, 0
  );

  CHECK(output_values[0] == 9.0f);
}

TEST_CASE("Conv2D input=1x1 filter=3x3 output=1x1 padding=same")
{
  Shape input_shape(1, 1, 1, 1);
  float input_values[1];
  std::fill_n(input_values, 1, 1.0f);

  Shape filter_shape(1, 3, 3, 1);
  float filter_values[9];
  std::fill_n(filter_values, 9, 1.0f);

  float bias_values[1] = { 0.0f, };

  Shape output_shape(1, 1, 1, 1);
  float output_values[1];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    1, 1
  );

  CHECK(output_values[0] == 1.0f);
}

TEST_CASE("Conv2D input=2x2 filter=3x3 output=2x2 padding=same")
{
  Shape input_shape(1, 2, 2, 1);
  float input_values[4];
  std::fill_n(input_values, 4, 1.0f);

  Shape filter_shape(1, 3, 3, 1);
  float filter_values[9];
  std::fill_n(filter_values, 9, 1.0f);

  float bias_values[1] = { 0.0f, };

  Shape output_shape(1, 2, 2, 1);
  float output_values[4];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    1, 1
  );

  float expected_output_values[4] = {
    4.0f, 4.0f,
    4.0f, 4.0f,
  };

  CHECK(std::equal(std::begin(output_values), std::end(output_values), std::begin(expected_output_values)));
}

TEST_CASE("Conv2D input=3x3 filter=3x3 output=3x3 padding=same")
{
  Shape input_shape(1, 3, 3, 1);
  float input_values[9];
  std::fill_n(input_values, 9, 1.0f);

  Shape filter_shape(1, 3, 3, 1);
  float filter_values[9];
  std::fill_n(filter_values, 9, 1.0f);

  float bias_values[1] = { 0.0f, };

  Shape output_shape(1, 3, 3, 1);
  float output_values[9];

  Conv2D(
    input_shape, input_values,
    filter_shape, filter_values,
    bias_values,
    output_shape, output_values,
    1, 1,
    1, 1
  );

  float expected_output_values[9] = {
    4.0f, 6.0f, 4.0f,
    6.0f, 9.0f, 6.0f,
    4.0f, 6.0f, 4.0f,
  };

  CHECK(std::equal(std::begin(output_values), std::end(output_values), std::begin(expected_output_values)));
}

