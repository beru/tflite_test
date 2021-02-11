project (tflite_test)

add_executable (tflite_test
    Minimal.cpp
    )

# For Tensorflow Lite
if (WIN32)
	target_link_libraries(tflite_test ${PROJECT_SOURCE_DIR}/tensorflowlite.dll.if.lib)
else()
	target_link_libraries(tflite_test ${PROJECT_SOURCE_DIR}/libtensorflowlite.so)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11 -lstdc++")
endif()

target_include_directories(tflite_test PUBLIC
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/tensorflow
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/absl
    ${PROJECT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/eigen_archive
    ${PROJECT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/gemmlowp
    ${PROJECT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/arm_neon_2_x86_sse
    ${PROJECT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/farmhash_archive/src
    ${PROJECT_SOURCE_DIR}/tensorflow/bazel-tensorflow/external/flatbuffers/include
)


