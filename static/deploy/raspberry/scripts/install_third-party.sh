# download third-part lib
if [ ! -d "./deps" ]; then
    mkdir deps
fi

if [ ! -d "./deps/gflags" ]; then
    cd deps
    git clone https://github.com/gflags/gflags
    cd gflags
    cmake .
    make -j 4
    cd ..
    cd ..
fi

# install yaml
YAML_URL=https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip
if [ ! -f "./deps/yaml-cpp.zip" ]; then
    cd deps
    wget -c ${YAML_URL}
    cd ..
fi

# install opencv
OPENCV_URL=https://bj.bcebos.com/paddlex/deploy/armlinux/opencv.tar.bz2
if [ ! -d "./deps/opencv" ]; then
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv.tar.bz2
    rm -rf opencv.tar.bz2
    cd ..
fi
