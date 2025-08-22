#!/bin/bash
# 1. 编译 CUDA 部分（使用你的 CUDA 11.2 路径）
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# 2. 编译 C++ 部分（适配 TF 2.5.0 + Python 3.8）
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include \  # TF 头文件路径
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include/external/nsync/public \  # TF 依赖的 nsync 头文件
-I /usr/local/cuda/include \  # CUDA 头文件路径
-lcudart \  # 链接 CUDA 运行时库
-L /usr/local/cuda/lib64/ \  # CUDA 库路径
-L /root/miniconda3/lib/python3.8/site-packages/tensorflow/ \  # TF 库路径（包含 libtensorflow_framework.so）
-ltensorflow_framework \  # 链接 TF 框架库（TF 2.x 必需）
-O2 \
-D_GLIBCXX_USE_CXX11_ABI=0  # 与 TF 编译时的 C++ ABI 保持一致（TF 2.5.0 默认用 0）