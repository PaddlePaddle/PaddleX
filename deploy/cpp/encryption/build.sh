if [ ! -d "3rd" ]; then
  mkdir 3rd
  cd 3rd

  wget https://bj.bcebos.com/paddlex/tools/openssl-1.1.0k.tar.gz
  tar -zxvf openssl-1.1.0k.tar.gz
  rm openssl-1.1.0k.tar.gz

  wget https://bj.bcebos.com/paddlex/deploy/gflags.tar.gz
  tar -zxvf gflags.tar.gz
  rm -rf gflags.tar.gz

  cd ..
fi

rm -rf build output
mkdir build && cd build

cmake ..
make -j16
