PADDLE_DIR=/path/to/paddle

if [ ! -d "3rd" ]; then
  mkdir 3rd
fi

cd 3rd
wget https://bj.bcebos.com/paddlex/tools/openssl-1.1.0k.tar.gz
tar -zxvf openssl-1.1.0k.tar.gz
rm openssl-1.1.0k.tar.gz

cd ..
rm -rf build output
mkdir build && cd build
cmake .. \
    -DPADDLE_DIR=${PADDLE_DIR}
make

