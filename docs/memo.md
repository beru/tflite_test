
# tflite::reference_integer_ops::ConvPerChannel について

`tensorflow/lite/kernels/internal/reference/integer_ops/conv.h` ファイル内の `ConvPerChannel` 関数

## 引数 output_multiplier と output_shift

スケーリングやバイアスの加算は出力チャンネル単位で行うので、内側の3重ループ（フィルタ縦横と入力チャンネル）で値を集積した後に行っている。

`MultiplyByQuantizedMultiplier` 関数を呼び出してスケーリング演算を行っている。`MultiplyByQuantizedMultiplier` 関数の実装は `tensorflow/lite/kernels/internal/common.h` ファイル内にあり、浮動小数点演算を使わず整数演算で行っているが、乗算と右シフトだけという単純なものでもないようだ。。

スケーリングを整数演算で行う方法については、`1712.05877.pdf` の `2.2. Integer-arithmetic-only matrix multiplication` に記載されている。

まず、式変形を行う事でスケーリングの乗算をまとめている。

式5 : M = (入力のスケール × カーネルのスケール) ÷ 出力のスケール

式6 : M = 2^-n * M0

事前に `M0(output_multiplier)` と `n(output_shift)` を計算している。

実装は、`tensorflow/lite/kernels/conv.cc` ファイル内の `tflite::ops::builtin::conv::Prepare` 関数で `tflite::PopulateConvolutionQuantizationParams` 関数を呼び出して行っている。

`tflite::PopulateConvolutionQuantizationParams` 関数は `tensorflow/lite/kernels/kernel_util.cc` ファイル内にあり、ループ処理でチャンネル毎に式5の計算を double 精度で行った後に `QuantizeMultiplier` 関数を読んで `significand` と `channel_shift` を求めている。`significand` が `output_multiplier` に記録されて、`channel_shift` が `output_shift` に記録される。

`QuantizeMultiplier` 関数は `tensorflow/lite/kernels/internal/quantization_util.cc` ファイル内に書かれている。

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

