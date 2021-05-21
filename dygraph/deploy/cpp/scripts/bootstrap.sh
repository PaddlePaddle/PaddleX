# download pre-compiled opencv lib
OPENCV_URL=https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2
{
    system_name=`awk -F= '/^NAME/{print $2}' /etc/os-release `
} || {
    echo "[ERROR] There's some problems, maybe caused by your system is not Ubuntu, refer this doc for more informat: https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/deploy/opencv.md"
    exit -1
}

# download pre-compiled opencv lib
OPENCV_URL=https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2
if [ $system_name == '"Ubuntu"' ]
then
    system_version=`awk -F= '/^VERSION_ID/{print $2}' /etc/os-release `
    if [ $system_version == '"18.04"' ]
    then
        OPENCV_URL=https://bj.bcebos.com/paddlex/deploy/opencv3.4.6gcc4.8ffmpeg_ubuntu_18.04.tar.gz2
    elif [ $system_version == '"16.04"' ]
    then
        OPENCV_URL=https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2
    else
        echo "[ERROR] Cannot find pre-comipled opencv lib for your system environment, refer this doc for more information: https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/deploy/opencv.md"
        exit -1
    fi
else
    echo "[ERROR] Cannot find pre-comipled opencv lib for your system environment, refer this doc for more information: https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/deploy/opencv.md"
    exit -1
fi

if [ ! -d "./deps/opencv3.4.6gcc4.8ffmpeg/" ]; then
    mkdir -p deps
    cd deps
    wget -c ${OPENCV_URL} -O opencv3.4.6gcc4.8ffmpeg.tar.gz2
    tar xvfj opencv3.4.6gcc4.8ffmpeg.tar.gz2
    rm -rf opencv3.4.6gcc4.8ffmpeg.tar.gz2
    cd ..
fi
