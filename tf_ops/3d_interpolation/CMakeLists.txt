cmake_minimum_required(VERSION 3.10)
project(tf_interpolate)

# 找到 TensorFlow 安装路径
if(NOT DEFINED TENSORFLOW_PATH)
  set(TENSORFLOW_PATH "/usr/local/lib/python3.8/site-packages/tensorflow")
endif()

# 找到 CUDA
find_package(CUDA REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# 包含 TensorFlow 头文件
include_directories(
  ${TENSORFLOW_PATH}/include
  ${CUDA_INCLUDE_DIRS}
)

# 链接 TensorFlow 库
link_directories(
  ${TENSORFLOW_PATH}
  ${TENSORFLOW_PATH}/lib
  ${CUDA_LIBRARY_DIRS}
)

# 编译自定义算子
cuda_add_library(tf_interpolate_so SHARED
  tf_interpolate.cpp
  tf_interpolate_kernel.cu
)

# 链接依赖
target_link_libraries(tf_interpolate_so
  tensorflow_framework
  ${CUDA_LIBRARIES}
)

# 设置输出文件名
set_target_properties(tf_interpolate_so PROPERTIES
  OUTPUT_NAME "tf_interpolate_so"
)