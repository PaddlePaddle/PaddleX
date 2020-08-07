# download pre-compiled paddle encrypt
ENCRYPTION_URL=https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip
if [ ! -d "./paddlex-encryption" ]; then
    wget -c ${ENCRYPTION_URL}
    unzip paddlex-encryption.zip
    rm -rf paddlex-encryption.zip
fi

# download pre-compiled opencv lib
OPENCV_URL=https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2
if [ ! -d "./deps/opencv3.4.6gcc4.8ffmpeg/" ]; then
    mkdir -p deps
    cd deps
    wget -c ${OPENCV_URL}
    tar xvfj opencv3.4.6gcc4.8ffmpeg.tar.gz2
    rm -rf opencv3.4.6gcc4.8ffmpeg.tar.gz2
    cd ..
fi
