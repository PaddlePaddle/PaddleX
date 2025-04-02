---
comments: true
---

# 昇腾 NPU 高性能推理教程

当前 PaddleX 高性能推理支持昇腾 910B、310P、310B 芯片（如果您有其他型号的相关需求，请提交issue告知我们）。考虑到环境差异性，我们推荐使用<b>飞桨官方提供的昇腾开发镜像</b>完成环境准备。

## 1、docker环境准备

* 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了昇腾算子库 CANN-8.0.0。

```bash
# 910B x86 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-x86_64-gcc84
# 910B aarch64 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-910b-base-aarch64-gcc84
# 310P aarch64 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-310p-base-aarch64-gcc84
# 310P x86 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-310p-base-x86_64-gcc84
# 310B aarch64 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-310b-base-aarch64-gcc84
# 310B x86 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann800-ubuntu20-npu-310b-base-x86_64-gcc84
```

* 以 910B x86 架构为例，使用如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号

```bash
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann80RC2-ubuntu20-npu-base-x86_64-gcc84 /bin/bash
```

## 2、安装PaddleX及高性能推理插件
### 2.1 安装PaddleX

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout develop
pip install -e .
```

### 2.2 安装高性能推理插件

高性能推理插件的 whl 包已经上传至 PaddleX 官方源，可以直接下载安装，也可以手动编译安装。

* 推荐直接下载安装 PaddleX 官方提供的 whl 包

```bash
# x86 架构
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/ultar_infer_whl/ultra_infer_python-1.0.0-cp310-cp310-linux_x86_64.whl
# aarch64 架构
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/ultar_infer_whl/ultra_infer_python-1.0.0-cp310-cp310-linux_aarch64.whl
```

* 手动编译安装

```bash
cd PaddleX/libs/ultra-infer/python
unset http_proxy https_proxy
# 使能om，onnx后端，禁用paddle后端，禁用gpu
export ENABLE_OM_BACKEND=ON ENABLE_ORT_BACKEND=ON ENABLE_PADDLE_BACKEND=OFF WITH_GPU=OFF
# 第一次编译需要设置
export ENABLE_VISION=ON
export ENABLE_TEXT=ON
# 注意，仅aarch机器需要设置NPU_HOST_LIB，指定libascend库
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
python setup.py build
python setup.py bdist_wheel
python -m pip install dist/ultra_infer_python*.whl
```

## 3、单模型推理

在昇腾上，PaddleX 高性能推理插件支持 OM 和 ORT 两种后端，其中 OM 后端基于 OM 模型，使用 npu 推理，性能更优；ORT 后端基于 ONNX 模型，使用 cpu 推理，支持模型较全，基本涵盖[PaddleX模型列表（昇腾 NPU）](../support_list/model_list_npu.md)所有模型。

OM 后端支持模型列表如下，不同芯片之间 OM 模型不通用（更多模型正在支持中，如果有需求的模型，可以提 issue 告知我们，也欢迎各位开发者提 pr 对贡献新模型）：

| 模型类型 | 模型名称 | 输入shape | 模型下载链接 |
| - | - | - | - |
| 文本检测 | PP-OCRv4_mobile_det | x:1,3,640,480 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_mobile_det/PP-OCRv4_mobile_det_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_mobile_det/PP-OCRv4_mobile_det_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_mobile_det/PP-OCRv4_mobile_det_infer_om_310B.tar) |
| 文本检测 | PP-OCRv4_server_det | x:1,3,640,480 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_server_det/PP-OCRv4_server_det_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_server_det/PP-OCRv4_server_det_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_server_det/PP-OCRv4_server_det_infer_om_310B.tar) |
| 文本识别 | PP-OCRv4_mobile_rec | x:1,3,48,320 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_mobile_rec/PP-OCRv4_mobile_rec_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_mobile_rec/PP-OCRv4_mobile_rec_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_mobile_rec/PP-OCRv4_mobile_rec_infer_om_310B.tar) |
| 文本识别 | PP-OCRv4_server_rec | x:1,3,48,320 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_server_rec/PP-OCRv4_server_rec_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_server_rec/PP-OCRv4_server_rec_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/PP-OCRv4_server_rec/PP-OCRv4_server_rec_infer_om_310B.tar) |
| 图像分类 | ResNet50 | x:1,3,224,224 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/ResNet50/ResNet50_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/ResNet50/ResNet50_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/ResNet50/ResNet50_infer_om_310B.tar) |
| 图像分类 | CLIP_vit_base_patch16_224 | x:1,3,224,224 | 待提供 |
| 图像多标签分类 | ResNet50_ML | x:1,3,448,448 | 待提供 |
| 目标检测 | RT-DETR-L | im_shape:1,2;image:1,3,640,640;scale_factor:1,2 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/RT-DETR-L/RT-DETR-L_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/RT-DETR-L/RT-DETR-L_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/RT-DETR-L/RT-DETR-L_infer_om_310B.tar) |
| 行人属性 | PP-LCNet_x1_0_pedestrian_attribute | x:1,3,256,192 | 待提供 |
| 车辆属性 | PP-LCNet_x1_0_vehicle_attribute | x:1,3,192,256 | 待提供 |
| 时序预测 | DLinear | past_target:1,96,1 | [910B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/DLinear/DLinear_infer_om_910B.tar)/[310P](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/DLinear/DLinear_infer_om_310P.tar)/[310B](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/infer_om/models/DLinear/DLinear_infer_om_310B.tar) |
| 时序异常检测 | DLinear_ad | observed_cov_numeric:1,96,2 | 待提供 |
| 图像特征 | PP-ShiTuV2_rec | x:1,3,224,224 | 待提供 |
| 印章文本检测 | PP-OCRv4_server_seal_det | x:1,3,640,480 | 待提供 |

### 3.1 OM后端推理

* 准备 OM 模型及配置文件

模型及配置文件命名需要固定为 inference.om 和 inference.yml，并放置在同级目录下，模型列表中提供的官方模型包含这两个文件。

如果您想使用自己的模型进行推理部署，先用 paddle2onnx 插件将导出的静态图模型转换为 ONNX 模型，再使用 ATC 工具转换为 OM 模型。inference.yml 在 PaddleX 导出模型时会自动生成，或者下载使用官方模型中的配置文件。

值得注意的是，目前只支持使用 OM 静态 shape 进行推理，在转换时需要指定 input_shape，各模型输入shape参考上表，如果指定 shape 推理精度异常，可以参考 PaddleX 导出模型生成的 inference.yml 配置文件，修改 input_shape 参数。动态 shape 会在下个版本支持。

以 PP-OCRv4_mobile_rec 为例说明转化方法：

 ```bash
# 先使用PaddleX提供的paddle2onnx插件将训练导出的静态图转成onnx模型
paddlex --paddle2onnx --paddle_model_dir <PaddlePaddle模型存储目录> --onnx_model_dir <ONNX模型存储目录>

# 昇腾默认支持fp16,即算子支持float16和float32数据类型时，强制选择float16
# 使用静态shape，通过参数input_shape指定输入shape
atc --model=inference.onnx --framework=5 --output=inference --soc_version=Ascend910B2 --input_shape "x:1,3,48,320"
# 如果需要fp32精度，需要在转换命令中加上--precision_mode_v2=origin
atc --model=inference.onnx --framework=5 --output=inference --soc_version=Ascend910B2 --input_shape "x:1,3,48,320" --precision_mode_v2=origin
```

更多关于 ATC 工具的使用，请参考[ATC工具学习向导](https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/Appendices/ttmutat/atctool_000003.html)

* 使用 PaddleX Python API 进行推理

以文本识别和图像分类为例进行说明。

PP-OCRv4_mobile_rec：

```python
# 下载推理示例图片
# wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png

from paddlex import create_model

hpi_config = {
    "auto_config": False,   # 关闭自动配置功能，手动配置后端
    "backend": "om", # 选用om后端
}
# model_name传入使用的模型名称，
# model_dir传入模型及配置文件存放的路径
# device设置为"npu:0"或"npu"，不设置卡号则默认使用0号卡
# use_hpip设置为True，开启高性能推理插件
# input_shape传入模型输入shape，以列表形式传入[c,w,h]，需要和atc转换时指定的输入shape保持一致，且目前只有OCR类的模型需要传入该参数
model = create_model(model_name="PP-OCRv4_mobile_rec", model_dir="PP-OCRv4_mobile_rec_infer_om_910b", device="npu:0", use_hpip=True, hpi_config=hpi_config, input_shape=[3, 48, 320])
output = model.predict("general_ocr_rec_001.png")
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

ResNet50：

```python
# 下载推理示例图片
# wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg

from paddlex import create_model

hpi_config = {
    "auto_config": False,   # 关闭自动配置功能，手动配置后端
    "backend": "om", # 选用om后端，昇腾上可选值为："onnxruntime"、"om"
}

# 无需传参input_shape
model = create_model(model_name="ResNet50", model_dir="ResNet50_infer_om_910b", device="npu", use_hpip=True, hpi_config=hpi_config)
output = model.predict("general_image_classification_001.jpg")
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

如遇到推理问题，可以先参考本文档第5小节：常见问题解决方法，如果仍未解决，可以给 PaddleX 官方提 issue，或者加入 PaddleX 官方交流群进行讨论。

### 3.2 ORT后端推理

ORT 后端推理使用方法与 OM 后端类似，OM 不支持的模型，可以使用 ORT 后端进行推理。

* 准备 ONNX 模型及配置文件
各模型的静态图权重可通过[PaddleX模型列表（昇腾 NPU）](../support_list/model_list_npu.md)进行下载，如果使用您自己训练的模型，可以使用 PaddleX 提供的 paddle2onnx 插件将静态图模型转换为 ONNX 模型，放置在指定目录下。

```bash
paddlex --paddle2onnx --paddle_model_dir <PaddlePaddle模型存储目录> --onnx_model_dir <ONNX模型存储目录>
```

* 使用 PaddleX Python API 进行推理

ORT 后端支持动态 shape，不需要考虑 input_shape 的问题；此外，需要将 hpi_config 中的 backend 改为 "onnxruntime" ，将 device 改为 "cpu"。

以 PP-OCRv4_mobile_rec 为例：

```python
# 下载推理示例图片
# wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png

from paddlex import create_model

hpi_config = {
    "auto_config": False,   # 关闭自动配置功能，手动配置后端
    "backend": "onnxruntime", # 选用onnxruntime后端
}

# device设置为"cpu"
# 无需设置input_shape
model = create_model(model_name="PP-OCRv4_mobile_rec", model_dir="PP-OCRv4_mobile_rec_infer_onnx", device="cpu", use_hpip=True, hpi_config=hpi_config)
output = model.predict("general_ocr_rec_001.png")
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

## 4、产线推理

高性能推理支持 PaddleX 产线推理，以 OCR 产线推理为例进行说明，关于 OCR 产线的详细介绍可以参考[通用OCR产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/OCR.md)。

* 准备 OM 模型及配置文件

类似单模型推理，先准备 PP-OCRv4_mobile_det 和 PP-OCRv4_mobile_rec 的om模型文件 inference.om 和配置文件 inference.yml，放在两个不同的目录下。

* 创建产线配置文件 OCR.yml

```yaml
# 在顶层设置hpi_config,指定推理后端为om
# 禁用om暂不支持的模块，主要保留检测和识别模块
# 在检测和识别模块中配置参数input_shape，设置静态shape
# 在检测和识别模块中配置参数model_dir，指向模型文件及配置文件的路径

pipeline_name: OCR

text_type: general

use_doc_preprocessor: False
use_textline_orientation: False

hpi_config:
  auto_config: False
  backend: om

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: False
    use_doc_unwarping: False
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null
      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null

SubModules:
  TextDetection:
    module_name: text_detection
    model_name: PP-OCRv4_mobile_det
    model_dir: PP-OCRv4_mobile_det_infer_om
    limit_side_len: 960
    limit_type: max
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 2.0
    input_shape: [3, 640, 480]
  TextLineOrientation:
    module_name: textline_orientation
    model_name: PP-LCNet_x0_25_textline_ori
    model_dir: null
    batch_size: 6
  TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv4_mobile_rec
    model_dir: PP-OCRv4_mobile_rec_infer_om
    batch_size: 1
    score_thresh: 0.0
    input_shape: [3, 48, 320]
```

* 使用 PaddleX Python API 进行推理

```python
# 下载推理示例图片
# wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png

from paddlex import create_pipeline
# pipeline设置为修改后的产线配置文件，use_hpip表示使用高性能推理
pipeline = create_pipeline(pipeline="./my_path/OCR.yaml", device="npu", use_hpip=True)

output = pipeline.predict(
    input="./general_ocr_002.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

需要注意的是，因为底层硬件的支持问题，在 arm 机器上，会出现 PP-OCRv4_mobile_det 推理卡住的问题，可以修改 OCR.yml 配置文件，将 PP-OCRv4_mobile_det 的推理后端设置为 onnxruntime，来规避这个问题。这个问题在后续版本中会修复。

修改 OCR.yml 如下：

```yaml
pipeline_name: OCR

text_type: general

use_doc_preprocessor: False
use_textline_orientation: False

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: False
    use_doc_unwarping: False
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null
      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null

# 在TextDetection配置hpi_config，指定后端为onnxruntime、设备为cpu，不指定输入shape
SubModules:
  TextDetection:
    module_name: text_detection
    model_name: PP-OCRv4_mobile_det
    model_dir: PP-OCRv4_mobile_det_infer_onnx
    limit_side_len: 960
    limit_type: max
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 2.0
    hpi_config:
      auto_config: False
      backend: onnxruntime
      device_type: cpu
  TextLineOrientation:
    module_name: textline_orientation
    model_name: PP-LCNet_x0_25_textline_ori
    model_dir: null
    batch_size: 6
  TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv4_mobile_rec
    model_dir: PP-OCRv4_mobile_rec_infer_om
    batch_size: 1
    score_thresh: 0.0
    hpi_config:
      auto_config: False
      backend: om
      device_type: npu
    input_shape: [3, 48, 320]
```

修改推理脚本如下：

```python
from paddlex import create_pipeline

# 不指定设备
pipeline = create_pipeline(pipeline="./my_path/OCR.yaml", use_hpip=True)

output = pipeline.predict(
    input="./general_ocr_002.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

## 5、常见问题解决方法
### 5.1 “RuntimeError: UltraInfer initalized failed! Error: libopencv_flann.so.3.4: cannot open shared object file: No such file or directory”

找不到 libopencv_flann.so.3.4 库，查找到该库在机器上的路径，然后将路径添加到 LD_LIBRARY_PATH 中，如：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/ultra_infer/libs/third_libs/opencv/lib:$LD_LIBRARY_PATH
```

### 5.2 “cannot allocate memory in static TLS block”

在 arm 机器上，可能会出现 “xxx.so cannot allocate memory in static TLS block” 的问题，查找报错的.so文件在机器上的路径，然后添加到 LD_PRELOAD 中，如：

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
export LD_PRELOAD=/usr/local/lib/python3.10/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
```
