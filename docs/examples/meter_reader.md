# Industrial meter readings

This case implements the detection and automatic reading of traditional mechanical pointer meters based on PaddleX, to open up the meter data and pre-training model, and provide the deployment guide for server-side on Windows-based systems and jetson embedded devices on Linux-based systems.

## Reading flow

Meter readings are completed in three steps:

* Step 1: Detect the meter in the image by using the object detection model.
* Step 2: Use the semantic segmentation model to segment the pointer and scales of each meter.
* Step 3: Calculate the reading of each meter based on the relative position of the pointer and the predicted range.

![MeterReader_Architecture](./images/MeterReader_Architecture.jpg)

* **Meter detection**: Since there is no meter with a small area in this case, the object detection model chooses **YOLOv3** that has better performance. With the consideration that this case is mainly deployed on devices with GPUs, **DarkNet53** with higher precision is chosen for the backbone network.
* **Scale and pointer segmentation**: With the consideration that the scale and pointer are in fine regions, the semantic segmentation model chooses **DeepLapv3** that has better performance.
* **Post-processing of readings**: 1. The semantically segmented prediction class map is subjected to an image etching operation for the purpose of scale segmentation. 2. The ring-shaped dial is expanded into a rectangular image, and a one-dimensional scale array and a one-dimensional pointer array are generated based on the class information in the image. 3. The mean value of the scale array is calculated, to use the mean value of the scale array for the binary operation. 4. The position of the pointer relative to the scale is located, to determine the type of dial according to the number of the scales to obtain the range of the dial, to multiply the relative position of the pointer and the range to get the reading of the dial.


## Metering data and pre-training models

This case opens up meter test images for experiencing the full flow of prediction inference for meter readings. It also opens up the meter detection dataset, and pointer and scale segmentation dataset to allow users to use these datasets for experiencing the training model.

| Meter Test Image | Meter Detection Dataset | Pointer and Scale Segmentation Dataset |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [meter_test](https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_test.tar.gz) | [meter_det](https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_det.tar.gz) | [meter_seg](https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_seg.tar.gz) |

The case opens up pre-trained detection and semantic segmentation models, which can be used to quickly experience the full flow of the meter reading, or deployed directly on server-side or jetson embedded devices to perform the inference prediction.

| Meter Detection Model | Pointer and Scale Segmentation Model |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [meter_det_inference_model](https://bj.bcebos.com/paddlex/examples/meter_reader/models/meter_det_inference_model.tar.gz) | [meter_seg_inference_model](https://bj.bcebos.com/paddlex/examples/meter_reader/models/meter_seg_inference_model.tar.gz) |


## Quick experience of dial readings

You can use the pre-trained model provided in this case to quickly experience the full flow of automatic prediction of meter readings. If you do not need a pre-trained model, you can go to the `model training` to restart the training model.

#### Pre-dependence

* Paddle paddle >= 1.8.0
* Python >= 3.5
* PaddleX >= 1.0.0

For installation related issues, refer to [PaddleX Installation]. (../install.md)

#### Test meter readings

Step 1. Download PaddleX source code:

```
git clone https://github.com/PaddlePaddle/PaddleX
```

Step 2. The prediction execution file is located in `PaddleX/examples/meter_reader/`. Access the directory:

```
cd PaddleX/examples/meter_reader/
```

The prediction execution file is `reader_infer.py`, and its main parameters are described as follows:


| Parameters | Description |
| ---- | ---- |
| detector_dir | Meter detection model path |
| segmenter_dir | Pointer and scale segmentation model paths |
| image | Image path to be predicted |
| image_dir | Folder paths to store images to be predicted |
| save_dir | Path to save the visualization results (default value is "output") |
| score_threshold | Boxes with prediction scores below this threshold are filtered out during the output of the detection model and the default value is 0.5 |
| seg_batch_size | Batch size of the segmentation (default value is 2) |
| use_camera | Whether to use a camera to capture pictures (default value is False) |
| camera_id | Camera device ID (default value is 0) |
| use_erode | Whether to subdivide the segmentation prediction images by using image erosion (default value is False) |
| erode_kernel | Convolution kernel size during image etching operation (default value is 4) |

Step 3. Prediction

If the GPU is used, the GPU card number is specified (for example, card 0):

```shell
export CUDA_VISIBLE_DEVICES=0
```
If the GPU is not used, CUDA_VISIBLE_DEVICES is set to null:
```shell
export CUDA_VISIBLE_DEVICES=
```

* Prediction of a single picture

```shell
python reader_infer.py --detector_dir /path/to/det_inference_model --segmenter_dir /path/to/seg_inference_model --image /path/to/meter_test/20190822_168.jpg --save_dir ./output --use_erode
```

* Prediction of multiple pictures

```shell
python reader_infer.py --detector_dir /path/to/det_inference_model --segmenter_dir /path/to/seg_inference_model --image_dir /path/to/meter_test --save_dir ./output --use_erode
```

* Start the camera for prediction

```shell
python reader_infer.py --detector_dir /path/to/det_inference_model --segmenter_dir /path/to/seg_inference_model --save_dir ./output --use_erode --use_camera
```

## Inference deployment

### Server-side security deployment of Windows systems

#### c++ deployment

Step 1. Download PaddleX source code:

```
git clone https://github.com/PaddlePaddle/PaddleX
```

Step 2. Copy the `meter_reader` folder and `CMakeList.txt` from `PaddleX\examples\meter_reader\deploy\cpp` to the `PaddleX\deploy\cpp` directory. Make a backup of the original `CMakeList.txt` in `PaddleX\deploy\cpp` before copying.

Step 3. Compile the C++ prediction code according to Step2 to Step4 in the [Windows platform deployment] (../deploy/server/cpp/windows.md).

Step 4. After successful compilation, the executable file is in the `out\build\x64-Release` directory. Run `cmd` and switch to that directory.

   ```
cd PaddleX\deploy\cpp\out\build\x64-Release
   ```

The prediction program is paddle_inference\meter_reader.exe. Its main command parameters are described below.

| Parameters | Description |
| ---- | ---- |
| det_model_dir | Meter detection model path |
| seg_model_dir | Pointer and scale segmentation model paths |
| image | Image path to be predicted |
| image_list | .txt file of storing image paths by line |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| gpu_id | GPU device ID (default value is 0) |
| save_dir | Path to save the visualization results (default value is "output") |
| seg_batch_size | Batch size of the segmentation (default value is 2) |
| thread_num | Number of threads for segmentation prediction (default value is the number of cpu processors) |
| use_camera | Whether to use a camera to capture pictures (value is 0 (default) or 1) |
| camera_id | Camera device ID (default value is 0) |
| use_erode | Whether to use image erosion to denoise the segmentation prediction images (value is 0 or 1 (default)) |
| erode_kernel | Convolution kernel size during image etching operation (default value is 4) |
| score_threshold | Boxes with prediction scores below this threshold are filtered out during the output of the detection model and the default value is 0.5|

Step 5. Inference prediction:

The model for the deployment inference should be in inference format. The pre-training models provided in this case are in inference format. For a re-trained model, you need to refer to the [Deployment Model Export](../deploy/export_model.md) to export the model to inference format.

* Use unencrypted models to make predictions on a single picture.

```shell
  .\paddlex_inference\meter_reader.exe --det_model_dir=\path\to\det_inference_model --seg_model_dir=\path\to\seg_inference_model --image=\path\to\meter_test\20190822_168.jpg --use_gpu=1 --use_erode=1 --save_dir=output
```

* Use unencrypted models to make predictions about image lists
The format of the image_list.txt content is as follows (it is not provided yet due to the different absolute paths, and can be generated by the user as required):
```
\path\to\images\1.jpg
\path\to\images\2.jpg
 . . .
\path\to\images\n.jpg
```

```shell
  .\paddlex_inference\meter_reader.exe --det_model_dir=\path\to\det_inference_model --seg_model_dir=\path\to\seg_inference_model --image_list=\path\to\meter_test\image_list.txt --use_gpu=1 --use_erode=1 --save_dir=output
```

* Use unencrypted models to start the camera to make predictions

```shell
  .\paddlex_inference\meter_reader.exe --det_model_dir=\path\to\det_inference_model --seg_model_dir=\path\to\seg_inference_model --use_camera=1 --use_gpu=1 --use_erode=1 --save_dir=output
```

* Prediction of a single image using an encrypted model

If the model is not encrypted, please refer to the [encrypted PaddleX model] to encrypt the model. For example, the directory where the encrypted detection model is located is `\path\to\encrypted_det_inference_model`, and the key is `yEBLDiBOdlj+5EsNNrABhfDuQGkdcreYcHcncqwdbx0=`; after encryption, the directory where the segmentation model is located is `\path\to\encrypted_seg_inference_model`, and the key is `DbVS64I9pFRo5XmQ8MNV2kSGsfEr4FKA6OH9OUhRrsY=`  

```shell
  .\paddlex_inference\meter_reader.exe --det_model_dir=\path\to\encrypted_det_inference_model --seg_model_dir=\path\to\encrypted_seg_inference_model --image=\path\to\test.jpg --use_gpu=1 --use_erode=1 --save_dir=output --det_key yEBLDiBOdlj+5EsNNrABhfDuQGkdcreYcHcncqwdbx0= --seg_key DbVS64I9pFRo5XmQ8MNV2kSGsfEr4FKA6OH9OUhRrsY=
```

### Security deployment of jetson embedded devices for Linux systems

#### c++ deployment

Step 1. Download PaddleX source code:

```
git clone https://github.com/PaddlePaddle/PaddleX
```

Step 2. Copy `meter_reader` folder and `CMakeList.txt` from `PaddleX/examples/meter_reader/deploy/cpp` to `PaddleX/deploy/cpp` directory. You can make a backup of the original `CMakeList.txt` in `PaddleX/deploy/cpp` before copying.

Step 3. Follow Step2 to Step3 in the [Deployment of Nvidia Jetson development panel](../deploy/nvidia-jetson.md) to compile the C++ prediction codes.

Step 4. After successful compilation, the executable program is `build/meter_reader/meter_reader`. Main command parameters are as follows:

| Parameters | Description |
| ---- | ---- |
| det_model_dir | Meter detection model path |
| seg_model_dir | Pointer and scale segmentation model paths |
| image | Image path to be predicted |
| image_list | .txt file of storing image paths by line |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| gpu_id | GPU device ID (default value is 0) |
| save_dir | Path to save the visualization results (default value is "output") |
| seg_batch_size | Batch size of the segmentation (default value is 2) |
| thread_num | Number of threads for segmentation prediction (default value is the number of cpu processors) |
| use_camera | Whether to use a camera to capture pictures (value is 0 (default) or 1) |
| camera_id | Camera device ID (default value is 0) |
| use_erode | Whether to subdivide the segmentation prediction pictures by using image erosion (value is 0 or 1 (default)) |
| erode_kernel | Convolution kernel size during image etching operation (default value is 4) |
| score_threshold | Boxes with prediction scores below this threshold are filtered out during the output of the detection model and the default value is 0.5|

Step 5. Inference prediction:

The model for the deployment inference should be in inference format. The pre-training models provided in this case are in inference format. For a re-trained model, you need to refer to the [Deployment Model Export](../deploy/export_model.md) to export the model to inference format.

* Use unencrypted models to make predictions on a single picture.

```shell
  ./build/meter_reader/meter_reader --det_model_dir=/path/to/det_inference_model --seg_model_dir=/path/to/seg_inference_model --image=/path/to/meter_test/20190822_168.jpg --use_gpu=1 --use_erode=1 --save_dir=output
```

* Use unencrypted models to make predictions about image lists
The format of the image_list.txt content is as follows (it is not provided yet due to the different absolute paths, and can be generated by the user as required):
```
\path\to\images\1.jpg
\path\to\images\2.jpg
. . .
\path\to\images\n.jpg
```
```shell
  ./build/meter_reader/meter_reader --det_model_dir=/path/to/det_inference_model --seg_model_dir=/path/to/seg_inference_model --image_list=/path/to/image_list.txt --use_gpu=1 --use_erode=1 --save_dir=output
```

* Use unencrypted models to start the camera to make predictions

```shell
  ./build/meter_reader/meter_reader --det_model_dir=/path/to/det_inference_model --seg_model_dir=/path/to/seg_inference_model --use_camera=1 --use_gpu=1 --use_erode=1 --save_dir=output
```

## Model training


#### Pre-dependence

* Paddle paddle >= 1.8.0
* Python >= 3.5
* PaddleX >= 1.0.0

For installation related issues, refer to [PaddleX Installation] (../install.md)

#### Training

* Training of dial detection
```
python /path/to/PaddleX/examples/meter_reader/train_detection.py
```
* Training of pointer and scale segmentation

```
python /path/to/PaddleX/examples/meter_reader/train_segmentation.py
```

Run the above script to train the detection model and segmentation model in this case. If you don't need the data and model parameters of this case, you can change the data, select the appropriate model and adjust the training parameters.
