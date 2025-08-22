#!/bin/bash
# 使用正确的CUDA路径（通过软链接指向11.2版本）
/usr/local/cuda/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# 适配TensorFlow 2.5.0和Python 3.8的编译命令
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include \
-I /usr/local/cuda/include \
-I /root/miniconda3/lib/python3.8/site-packages/tensorflow/include/external/nsync/public \
-lcudart \
-L /usr/local/cuda/lib64/ \
-L /root/miniconda3/lib/python3.8/site-packages/tensorflow \
-ltensorflow_framework \
-O2 \
-D_GLIBCXX_USE_CXX11_ABI=0