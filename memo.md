## TensorFlow Lite を Windows でビルドする方法について

https://www.tensorflow.org/install/source_windows

の記述通りに準備すればビルドが出来る。

https://qiita.com/na0ki_ikeda/items/4c34db69fc6c9f16673a

に書かれていた下記の記述でビルドが出来る。

```
bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite:tensorflowlite

```

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 16 2019" -A x64 ..
```

## How to build TensorFlow Lite on Linux

https://www.tensorflow.org/install/source#setup_for_linux_and_macos

Follow the instructions.

After `./configure`, you can build `libtensorflowlite.so` with below command.

```
bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite:tensorflowlite
```

You may find the library file inside `bazel-bin/tensorflow/lite` folder.

### gcc CLI build

https://qiita.com/iwatake2222/items/d998df1981d46285df62

```
gcc Classification.cpp -I. -I./tensorflow -I./tensorflow/lite/tools/make/downloads -I./tensorflow/lite/tools/make/downloads/eigen -I./tensorflow/lite/tools/make/downloads/absl -I./tensorflow/lite/tools/make/downloads/gemmlowp -I./tensorlow/lite/tools/make/downloads/neon_2_sse -I./tensorflow/lite/tools/make/downloads/farmhash/src -I./tensorflow/lite/tools/make/downloads/flatbuffers/include  -std=c++11 -lstdc++ -ltensorflowlite -lm -L./
```

## How to use BuiltinRefOpResolver insted of BuiltinOpResolver

modify `tensorflow/lite/BUILD` file,
add below line to `deps` attribute of `tflite_cc_shared_object` rule.
```
        "//tensorflow/lite/kernels:reference_ops",
```

then, build the library with bazel.

## Bazel

https://github.com/bazelbuild/bazel/releases

## EfficientNet

https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2

## ImageNet labels

https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

## flatc

https://github.com/google/flatbuffers/releases/download/v1.12.0/flatc_windows.zip

