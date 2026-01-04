export ENABLE_ORT_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_TRT_BACKEND=ON
export TRT_DIRECTORY=/ultra_infer/libs/TensorRT-8.4.1.5
export CUDA_DIRECTORY=/usr/local/cuda
export ENABLE_VISION=ON
export WITH_GPU=ON
export CMAKE_CXX_COMPILER=/usr/local/gcc-8.2/bin/g++

python setup.py build
python setup.py bdist_wheel
