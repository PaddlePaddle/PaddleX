# download pre-compiled opencv lib
OPENCV_URL=https://bj.bcebos.com/paddlex/deploy/tools/opencv3_aarch.tgz
if [ ! -d "./deps/opencv3" ]; then
    mkdir -p deps
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv3_aarch.tgz
    rm -rf opencv3_aarch.tgz
    cd ..
fi
