
## TensorFlow Lite を Windows でビルドする方法について

https://www.tensorflow.org/install/source_windows

の記述通りに準備すればビルドが出来る。

https://qiita.com/na0ki_ikeda/items/4c34db69fc6c9f16673a

に書かれていた下記の記述でビルドが出来る。

```
bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite:tensorflowlite
```

`BuiltinOpResolver` の代わりに `BuiltinRefOpResolver` を使う場合は、
`tensorflow/lite/BUILD` ファイルの `tflite_cc_shared_object` の `deps` に
```
        "//tensorflow/lite/kernels:reference_ops",
```
を追加してからビルドする。


