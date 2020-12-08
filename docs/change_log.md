# Update log

**v1.2.0** 2020.09.07
- Model Update
   > - Add the most practical object detection model PP-YOLO in the industry. Deeply considering the double requirements for precision and speed in the industrial application, the COCO dataset precision is 45.2% and the Tesla V100 inference speed is 72.9 FPS. [Details link] (https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo)
   > - Add to FasterRCNN, MaskRCNN, YOLOv3, DeepLabv3p and other models a built-in COCO dataset pre-training model which applies to fine-tuned training of small datasets. 
   > - Add to object detection models FasterRCNN and MaskRCNN backbone HRNet_W18 which applies to application scenarios having high requirements for details inference. [Details link] (https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn)
   > - Add backbone MobileNetV3_large_ssld to the semantic segmentation model DeepLabv3p. The model volume is 9.3 MB and the Cityscapes dataset precision still is 73.28%. [Details link] (https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p)

- Model Deployment Update
   > - Add a model inference acceleration deployment solution via OpenVINO. Compared with the mkldnn acceleration library, the inference speed increases by about 1.5-2 times on the CPU. [Details link] (https://paddlex.readthedocs.io/zh_CN/develop/deploy/openvino/index.html)
   > - Add a model deployment solution on Raspberry Pi and further enrich an edge deployment solution. [Details link] (https://paddlex.readthedocs.io/zh_CN/develop/deploy/raspberry/index.html)
   > - Optimize the data preprocessing and postprocessing code performance of PaddleLite Android deployment. The preprocessing speed increases by about 10 times and the postprocessing speed increases by about 4 times.
   > - Optimize C++ deployment codes on the Paddle server and add parameters such as use_mkl. Compared with not starting mkldnn, the inference speed increases by about 10-50 times on the CPU.


- Industrial Case Update
   > - Add a remote sensing segmentation case of large RGB images and provide a sliding window inference API, which can not only avoid the occurrence of insufficient GPU memory, but also eliminate the cracking feeling at the splice of the windows in the final inference results by configuring the degree of overlapping. [Details link](https://paddlex.readthedocs.io/zh_CN/develop/examples/remote_sensing.html)
   > - Add a multi-channel remote sensing image segmentation case and bridge the whole process of data analysis, model training and model deployment of semantic segmentation tasks on any number of channels. [Details link](https://paddlex.readthedocs.io/zh_CN/develop/examples/multi-channel_remote_sensing/README.html)


- Others
   > - Add the dataset splitting function which supports splitting ImageNet, PascalVOC, MSCOCO and semantic segmentation datasets with one click via command line. [Details link] (https://paddlex.readthedocs.io/zh_CN/develop/data/format/classification.html#id2)


**v1.1.0** 2020.07.12

- Model Update
> - Add semantic segmentation models HRNet and FastSCNN
> - Add backbone HRNet to the object detection FasterRCNN and the instance segmentation MaskRCNN
> - Add a COCO dataset pre-training model to the object detection/ instance segmentation model
> - Integrate X2Paddle. All PaddleX classification and semantic segmentation models support export as an ONNX protocol
- Model Deployment Update
> - Add the support for the Windows platform in model encryption
> - Add a Jetson and Paddle Lite model deployment and inference solution
> - Add batch inference in the C++ deployment codes and use OpenMP for parallel acceleration of preprocessing
- Add two PaddleX Industrial Cases
> - [Portrait segmentation case] (https://paddlex.readthedocs.io/zh_CN/develop/examples/human_segmentation.html)
> - [Industrial instrument reading case] (https://paddlex.readthedocs.io/zh_CN/develop/examples/meter_reader.html)
- Add the data format conversion function which converts data annotated by LabelMe, Colabeler and the EasyData platform into a data format that PaddleX supports loading
- Update the PaddleX document by optimizing the document structure


**v1.0.0** 2020.05.20

- Add model C++ and Python deployment codes
- Add a model encryption deployment solution
- Add an OpenVINO deployment solution for classification models
- Add a model interpretability API


**v0.1.8** 2020.05.17

- Fix some code bugs
- Add the support for the data annotation format on the EasyData platform
- Support the pixel-level operator in the imgaug data enhancement library
