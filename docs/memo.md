
# 


# Netronの表示について

real_value = scale * (quantized_value - zero_point)

Netron で Quantize されたモデルの input や output を見ると例えば下記のような表示になっている。

```
name: images_int8
type: int8[1,224,224,3]
quantization: -1.6488090753555298 ≤ 0.012566016986966133 * (q - 3) ≤ 1.5555250644683838
location: 0
```

int8 の最大値と最小値に zero_point を引くと下記の値になる

```
 127 - 3 = 124
-128 - 3 = -131
```

そしてその値に scale を掛けると下記の値になる。

```
0.012566016986966133 *  124 = 1.558186106383800492
0.012566016986966133 * -131 = -1.646148225292563423
```

```
quantization: -1.6488090753555298 ≤ 0.012566016986966133 * (q - 3) ≤ 1.5555250644683838
```

というのは、

```
quantization: min ≤ scale * (q - zero_point) ≤ max
```

を出力したもの。Netron のソースコード（tflite.js）の文字列化した箇所を確認した。

Netron ではこの min と max が見れるが、Python や C++ で TFLite を使ってもこの情報を見ることは出来ないようだ。

Netron のソースコードの `tflite-schema.js` を見る感じでは flatc で `tflite.schema` を使って生成したものなのかもしれない。

`tensorflow/lite/schema` には `schema.fbs` から生成したと思われる `schema_generated.h` ファイル中の構造体には min と max のメンバーが存在するが、

```
struct QuantizationParametersT : public flatbuffers::NativeTable {
  typedef QuantizationParameters TableType;
  std::vector<float> min;
  std::vector<float> max;
  std::vector<float> scale;
  std::vector<int64_t> zero_point;
  tflite::QuantizationDetailsUnion details;
  int32_t quantized_dimension;
  QuantizationParametersT()
      : quantized_dimension(0) {
  }
};
```

`tensorflow/lite/c/common.h` 中の `TfLiteAffineQuantization` にはそれに相当するメンバーが無い。

```
typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray* scale;
  TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;
```

`interpreter_builder.cc` の `InterpreterBuilder::ParseQuantization` で `TfLiteAffineQuantization` への記録を行っているが、min と max については記録先がそもそも無いので記録しようがない。

# 量子化

Quantization and Training of Neural Networks for EfficientInteger-Arithmetic-Only Inference
https://arxiv.org/pdf/1712.05877.pdf

TensorFlow Lite 8-bit quantization specification
https://www.tensorflow.org/lite/performance/quantization_spec?hl=en

