project (tflite_test)

add_executable (tflite_test
    Classification.cpp
    )

# For Tensorflow Lite
target_link_libraries(tflite_test ${PROJECT_SOURCE_DIR}/libtensorflowlite.so)
target_include_directories(tflite_test PUBLIC
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/tensorflow
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/eigen
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/absl
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/gemmlowp
    ${PROJECT_SOURCE_DIR}/tensorlow/lite/tools/make/downloads/neon_2_sse
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/farmhash/src
    ${PROJECT_SOURCE_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11 -lstdc++")

