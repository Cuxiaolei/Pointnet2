#!/bin/bash
# 编译 CUDA 部分
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#
# 编译 C++ 部分（注意每行末尾的反斜杠和无空格）
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include/external/nsync/public \
-I /usr/local/cuda/include \
-lcudart \
-L /usr/local/cuda/lib64/ \
-L /root/miniconda3/lib/python3.8/site-packages/tensorflow/ \
-ltensorflow_framework \
-O2 \
-D_GLIBCXX_USE_CXX11_ABI=0