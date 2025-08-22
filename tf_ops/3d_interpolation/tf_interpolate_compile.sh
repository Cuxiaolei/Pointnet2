#!/bin/bash
# 编译CUDA部分（如果有.cu文件需要先编译，此处假设原脚本可能遗漏，补充完整流程）
/usr/local/cuda/bin/nvcc tf_interpolate_g.cu -o tf_interpolate_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# 适配TensorFlow 2.x和当前环境的编译命令
g++ -std=c++11 tf_interpolate.cpp tf_interpolate_g.cu.o -o tf_interpolate_so.so -shared -fPIC \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include/external/nsync/public \
-I /usr/local/cuda/include \
-lcudart \
-L /usr/local/cuda/lib64/ \
-L /root/miniconda3/lib/python3.8/site-packages/tensorflow/ \
-ltensorflow_framework \
-O2 \
-D_GLIBCXX_USE_CXX11_ABI=0