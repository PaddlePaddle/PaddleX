# Deployment compilation description

Currently, the deployment test environment of all models in the PaddleX is
- Ubuntu 16.04/18.04 / Windows 10
- gcc 4.8.5 / Microsoft Visual Studio 2019

If you switch to another Linux environment (for the gcc version, it remains unchanged), the opencv problem may occur.

In the Linux compiling scripts, for example, `deploy/cpp/script/build.sh`, it depends on `deploy/cpp/script/bootstrap.sh` to automatically download pre-compiled dependencies of the opencv library and crypto library. Currently, `bootstrap.sh` provides only pre-compiling packages for OpenCV in two system environments in Ubuntu16.04/18.04. If your system is different from this, try the solution in the following method:


## Compiling of OpenCV in the Linux

### 1. Download the OpenCV Source Code
Go to the OpenCV official website and download the OpenCV 3.4.6 Source Code, or [click here](https://bj.bcebos.com/paddlex/deploy/opencv-3.4.6.zip) to download the source code compression package uploaded to the server.

### 2. Compile OpenCV.
Make sure your gcc/g++ version is 4.8.5. In the compilation process, use the following codes for reference.

The current path of opencv-3.4.6.zip is `/home/paddlex/opencv-3.4.6.zip`.
```
unzip opencv-3.4.6.zip
cd opencv-3.4.6
mkdir build && cd build
mkdir opencv3.4.6gcc4.8ffmpeg
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/home/paddlex/opencv-3.4.6/build/opencv3.4.6gcc4.8ffmpeg -D WITH_FFMPEG=ON ..
make -j5
make install
```
The compiled opencv is stored in the `/home/paddlex/opencv-3.4.6/build/opencv3.4.6gcc4.8ffmpeg` setting.

### 3. Compile the PaddleX prediction code with the dependency of the opencv.
Modify `deploy/cpp/script/build.sh`.

1. Note out or delete the following code:

```
{ 
    bash $(pwd)/scripts/bootstrap.sh # Download pre-compiled versions of encryption tools and opencv dependencies library 
}| {
    echo "Fail to execute script/bootstrap.sh"
    exit -1
}
```

2. Set the model encryption switch
If you do not need to use PaddleX's model encryption feature, set WITH_ENCRYPTION to OFF
```
WITH_ENCRYPTION=OFF
```
If you need to use encryption, download the encryption library manually and unzip it, [click to download](https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip).

3. Set the path of the Dependency Library
Set `OPENCV_DIR` to its the compiled path. For example,
```
OPENCV_DIR=/home/paddlex/opencv-3.4.6/build/opencv3.4.6gcc4.8ffmpeg
```
If you also need to use the model for encryption, set `WITH_ENCRYPTION` to `ON` and set `ENCRYPTION_DIR` to the path where you download and decompress.
```
ENCRYPTION_DIR=/home/paddlex/paddlex-encryption
```

4. Run `sh script/build.sh` to compile it.

## Feedback

If you still have a problem, please go to PaddleX's Github and submit your ISSUE for feedback.

- [PaddleX Issue](https://github.com/PaddlePaddle/PaddleX/issues)
