---
comments: true
---

# General Object Detection Pipeline Tutorial

## 1. Introduction to General Object Detection Pipeline
Object detection aims to identify the categories and locations of multiple objects in images or videos by generating bounding boxes to mark these objects. Unlike simple image classification, object detection not only requires recognizing what objects are present in an image, such as people, cars, and animals, but also accurately determining the specific position of each object within the image, typically represented by rectangular boxes. This technology is widely used in autonomous driving, surveillance systems, smart photo albums, and other fields, relying on deep learning models (e.g., YOLO, Faster R-CNN) that can efficiently extract features and perform real-time detection, significantly enhancing the computer's ability to understand image content.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/01.png">



<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PicoDet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>16.6715</td>
<td>169.904</td>
<td>20.9 M</td>
<td rowspan="2">PP-PicoDet is a lightweight object detection algorithm for full-size, wide-angle targets, considering the computational capacity of mobile devices. Compared to traditional object detection algorithms, PP-PicoDet has a smaller model size and lower computational complexity, achieving higher speed and lower latency while maintaining detection accuracy.</td>
</tr>
<tr>
<td>PicoDet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Trained Model</a></td>
<td>29.1</td>
<td>14.097</td>
<td>37.6563</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Trained Model</a></td>
<td>52.9</td>
<td>33.5644</td>
<td>814.825</td>
<td>185.3 M</td>
<td rowspan="2">PP-YOLOE_plus is an upgraded version of the high-precision cloud-edge integrated model PP-YOLOE, developed by Baidu's PaddlePaddle vision team. By using the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the model's end-to-end inference speed.</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Trained Model</a></td>
<td>43.7</td>
<td>16.8884</td>
<td>223.059</td>
<td>28.3 M</td>
</tr>
<tr>
<td>RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>56.3</td>
<td>114.814</td>
<td>3933.39</td>
<td>435.8 M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detector. The model features an efficient hybrid encoder to meet both model performance and throughput requirements, efficiently handling multi-scale features, and proposes an accelerated and optimized query selection mechanism to optimize the dynamics of decoder queries. RT-DETR supports flexible end-to-end inference speeds by using different decoders.</td>
</tr>
<tr>
<td>RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>53.0</td>
<td>34.5252</td>
<td>1454.27</td>
<td>113.7 M</td>
</tr>
</table>

> ❗ The above list features the <b>6 core models</b> that the image classification module primarily supports. In total, this module supports <b>37 models</b>. The complete list of models is as follows:

<details><summary> 👉Details of Model List</summary>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>41.1</td>
<td>-</td>
<td>-</td>
<td>245.4 M</td>
<td rowspan="2">Cascade-FasterRCNN is an improved version of the Faster R-CNN object detection model. By coupling multiple detectors and optimizing detection results using different IoU thresholds, it addresses the mismatch problem between training and prediction stages, enhancing the accuracy of object detection.</td>
</tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Trained Model</a></td>
<td>45.0</td>
<td>-</td>
<td>-</td>
<td>246.2 M</td>
</tr>
<tr>
<td>CenterNet-DLA-34</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-DLA-34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-DLA-34_pretrained.pdparams">Trained Model</a></td>
<td>37.6</td>
<td>-</td>
<td>-</td>
<td>75.4 M</td>
<td rowspan="2">CenterNet is an anchor-free object detection model that treats the keypoints of the object to be detected as a single point—the center point of its bounding box, and performs regression through these keypoints.</td>
</tr>
<tr>
<td>CenterNet-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>38.9</td>
<td>-</td>
<td>-</td>
<td>319.7 M</td>
</tr>
<tr>
<td>DETR-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DETR-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DETR-R50_pretrained.pdparams">Trained Model</a></td>
<td>42.3</td>
<td>59.2132</td>
<td>5334.52</td>
<td>159.3 M</td>
<td>DETR is a transformer-based object detection model proposed by Facebook. It achieves end-to-end object detection without the need for predefined anchor boxes or NMS post-processing strategies.</td>
</tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">Trained Model</a></td>
<td>37.8</td>
<td>-</td>
<td>-</td>
<td>137.5 M</td>
<td rowspan="9">Faster R-CNN is a typical two-stage object detection model that first generates region proposals and then performs classification and regression on these proposals. Compared to its predecessors R-CNN and Fast R-CNN, Faster R-CNN's main improvement lies in the region proposal aspect, using a Region Proposal Network (RPN) to provide region proposals instead of traditional selective search. RPN is a Convolutional Neural Network (CNN) that shares convolutional features with the detection network, reducing the computational overhead of region proposals.</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>38.4</td>
<td>-</td>
<td>-</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>39.5</td>
<td>-</td>
<td>-</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Trained Model</a></td>
<td>41.4</td>
<td>-</td>
<td>-</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>36.7</td>
<td>-</td>
<td>-</td>
<td>120.2 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101-FPN_pretrained.pdparams">Trained Model</a></td>
<td>41.4</td>
<td>-</td>
<td>-</td>
<td>216.3 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101_pretrained.pdparams">Trained Model</a></td>
<td>39.0</td>
<td>-</td>
<td>-</td>
<td>188.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNeXt101-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>43.4</td>
<td>-</td>
<td>-</td>
<td>360.6 M</td>
</tr>
<tr>
<td>FasterRCNN-Swin-Tiny-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-Swin-Tiny-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-Swin-Tiny-FPN_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>-</td>
<td>-</td>
<td>159.8 M</td>
</tr>
<tr>
<td>FCOS-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>39.6</td>
<td>103.367</td>
<td>3424.91</td>
<td>124.2 M</td>
<td>FCOS is an anchor-free object detection model that performs dense predictions. It uses the backbone of RetinaNet and directly regresses the width and height of the target object on the feature map, predicting the object's category and centerness (the degree of offset of pixels on the feature map from the object's center), which is eventually used as a weight to adjust the object score.</td>
</tr>
<tr>
<td>PicoDet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>16.6715</td>
<td>169.904</td>
<td>20.9 M</td>
<td rowspan="4">PP-PicoDet is a lightweight object detection algorithm designed for full-size and wide-aspect-ratio targets, with a focus on mobile device computation. Compared to traditional object detection algorithms, PP-PicoDet boasts smaller model sizes and lower computational complexity, achieving higher speeds and lower latency while maintaining detection accuracy.</td>
</tr>
<tr>
<td>PicoDet-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">Trained Model</a></td>
<td>37.5</td>
<td>16.2311</td>
<td>71.7257</td>
<td>16.8 M</td>
</tr>
<tr>
<td>PicoDet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Trained Model</a></td>
<td>29.1</td>
<td>14.097</td>
<td>37.6563</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PicoDet-XS</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">Trained Model</a></td>
<td>26.2</td>
<td>13.8102</td>
<td>48.3139</td>
<td>5.7 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Trained Model</a></td>
<td>52.9</td>
<td>33.5644</td>
<td>814.825</td>
<td>185.3 M</td>
<td rowspan="4">PP-YOLOE_plus is an iteratively optimized and upgraded version of PP-YOLOE, a high-precision cloud-edge integrated model developed by Baidu PaddlePaddle's Vision Team. By leveraging the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the end-to-end inference speed of the model.</td>
</tr>
<tr>
<td>PP-YOLOE_plus-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">Trained Model</a></td>
<td>49.8</td>
<td>19.843</td>
<td>449.261</td>
<td>82.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Trained Model</a></td>
<td>43.7</td>
<td>16.8884</td>
<td>223.059</td>
<td>28.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">Trained Model</a></td>
<td>54.7</td>
<td>57.8995</td>
<td>1439.93</td>
<td>349.4 M</td>
</tr>
<tr>
<td>RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>56.3</td>
<td>114.814</td>
<td>3933.39</td>
<td>435.8 M</td>
<td rowspan="5">RT-DETR is the first real-time end-to-end object detector. It features an efficient hybrid encoder that balances model performance and throughput, efficiently processes multi-scale features, and introduces an accelerated and optimized query selection mechanism to dynamize decoder queries. RT-DETR supports flexible end-to-end inference speeds through the use of different decoders.</td>
</tr>
<tr>
<td>RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>53.0</td>
<td>34.5252</td>
<td>1454.27</td>
<td>113.7 M</td>
</tr>
<tr>
<td>RT-DETR-R18</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">Trained Model</a></td>
<td>46.5</td>
<td>19.89</td>
<td>784.824</td>
<td>70.7 M</td>
</tr>
<tr>
<td>RT-DETR-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R50_pretrained.pdparams">Trained Model</a></td>
<td>53.1</td>
<td>41.9327</td>
<td>1625.95</td>
<td>149.1 M</td>
</tr>
<tr>
<td>RT-DETR-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-X_pretrained.pdparams">Trained Model</a></td>
<td>54.8</td>
<td>61.8042</td>
<td>2246.64</td>
<td>232.9 M</td>
</tr>
<tr>
<td>YOLOv3-DarkNet53</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">Trained Model</a></td>
<td>39.1</td>
<td>40.1055</td>
<td>883.041</td>
<td>219.7 M</td>
<td rowspan="3">YOLOv3 is a real-time end-to-end object detector that utilizes a unique single Convolutional Neural Network (CNN) to frame the object detection problem as a regression task, enabling real-time detection. The model employs multi-scale detection to enhance performance across different object sizes.</td>
</tr>
<tr>
<td>YOLOv3-MobileNetV3</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-MobileNetV3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-MobileNetV3_pretrained.pdparams">Trained Model</a></td>
<td>31.4</td>
<td>18.6692</td>
<td>267.214</td>
<td>83.8 M</td>
</tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-ResNet50_vd_DCN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-ResNet50_vd_DCN_pretrained.pdparams">Trained Model</a></td>
<td>40.6</td>
<td>31.6276</td>
<td>856.047</td>
<td>163.0 M</td>
</tr>
<tr>
<td>YOLOX-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-L_pretrained.pdparams">Trained Model</a></td>
<td>50.1</td>
<td>185.691</td>
<td>1250.58</td>
<td>192.5 M</td>
<td rowspan="6">Building upon YOLOv3's framework, YOLOX significantly boosts detection performance in complex scenarios by incorporating Decoupled Head, Data Augmentation, Anchor Free, and SimOTA components.</td>
</tr>
<tr>
<td>YOLOX-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-M_pretrained.pdparams">Trained Model</a></td>
<td>46.9</td>
<td>123.324</td>
<td>688.071</td>
<td>90.0 M</td>
</tr>
<tr>
<td>YOLOX-N</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">Trained Model</a></td>
<td>26.1</td>
<td>79.1665</td>
<td>155.59</td>
<td>3.4 M</td>
</tr>
<tr>
<td>YOLOX-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-S_pretrained.pdparams">Trained Model</a></td>
<td>40.4</td>
<td>184.828</td>
<td>474.446</td>
<td>32.0 M</td>
</tr>
<tr>
<td>YOLOX-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-T_pretrained.pdparams">Trained Model</a></td>
<td>32.9</td>
<td>102.748</td>
<td>212.52</td>
<td>18.1 M</td>
</tr>
<tr>
<td>YOLOX-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-X_pretrained.pdparams">Trained Model</a></td>
<td>51.8</td>
<td>227.361</td>
<td>2067.84</td>
<td>351.5 M</td>
</tr>
</table>

<p><b>Note: The precision metrics mentioned are based on the <a href="https://cocodataset.org/#home">COCO2017</a> validation set mAP(0.5:0.95). All model GPU inference times are measured on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p></details>

## 2. Quick Start
PaddleX's pre-trained model pipelines allow for quick experience of their effects. You can experience the effects of the General Object Detection Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience the General Object Detection Pipeline online](https://aistudio.baidu.com/community/app/70230/webUI) using the demo images provided by the official source, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/02.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model within the pipeline</b>.

### 2.2 Local Experience
Before using the General Object Detection Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.md).

#### 2.2.1 Command Line Experience
A single command can quickly experience the effects of the object detection pipeline, Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline object_detection --input general_object_detection_002.png --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the object detection pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default object detection pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> Click to expand 👉</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config object_detection
</code></pre>
<p>After execution, the object detection pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config object_detection --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, replace <code>--pipeline</code> with the configuration file save path to make the configuration file effective. For example, if the configuration file save path is <code>./object_detection.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./object_detection.yaml --input general_object_detection_002.png --device gpu:0
</code></pre>
<p>Here, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If these parameters are still specified, the specified parameters will take precedence.</p></details>

After running, the result will be:

```
{'input_path': 'general_object_detection_002.png', 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188097476959229, 'coordinate': [661, 93, 870, 305]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7743489146232605, 'coordinate': [76, 274, 330, 520]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7270504236221313, 'coordinate': [285, 94, 469, 297]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5570532083511353, 'coordinate': [310, 361, 685, 712]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5484835505485535, 'coordinate': [764, 285, 924, 440]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5160726308822632, 'coordinate': [853, 169, 987, 303]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.5142655968666077, 'coordinate': [0, 0, 1072, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5101479291915894, 'coordinate': [57, 23, 213, 176]}]}
```

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/03.png">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

#### 2.2.2 Integration via Python Scripts
A few lines of code are all you need to quickly perform inference on your production line. Taking General Object Detection as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="object_detection")

output = pipeline.predict("general_object_detection_002.png")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the visualized image of the result
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference, only available if the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the pipeline object to perform inference: The `predict` method parameter `x` is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing a dictionary type, where the key needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above data types, e.g., <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing a list, where the list elements need to be of the above types, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>Prints results to the terminal</td>
<td><code>- format_json</code>: bool, whether to format the output content with json indentation, default is True;<br/><code>- indent</code>: int, json formatting setting, only valid when format_json is True, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, only valid when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Saves results as a json file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br/><code>- indent</code>: int, json formatting setting, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Saves results as an image file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;</td>
</tr>
</tbody>
</table>
If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/object_detection.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
output = pipeline.predict("general_object_detection_002.png")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualized image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. Refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md) for detailed high-performance inference procedures.

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>

<p>For primary operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and the response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>Primary operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Performs object detection on an image.</p>
<p><code>POST /object-detection</code></p>
<ul>
<li>The request body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of an image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>Information about the location and category of the detected objects.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The image of the object detection result. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>detectedObjects</code> is an <code>object</code> with the following properties:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the object. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively.</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>The ID of the object category.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The score of the object.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;detectedObjects&quot;: [
{
&quot;bbox&quot;: [
404.4967956542969,
90.15770721435547,
506.2465515136719,
285.4187316894531
],
&quot;categoryId&quot;: 0,
&quot;score&quot;: 0.7418514490127563
},
{
&quot;bbox&quot;: [
155.33145141601562,
81.10954284667969,
199.71136474609375,
167.4235382080078
],
&quot;categoryId&quot;: 1,
&quot;score&quot;: 0.7328268885612488
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/object-detection&quot;
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected objects:&quot;)
print(result[&quot;detectedObjects&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;image&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    auto response = client.Post(&quot;/object-detection&quot;, headers, body, &quot;application/json&quot;);
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto detectedObjects = result[&quot;detectedObjects&quot;];
        std::cout &lt;&lt; &quot;\nDetected objects:&quot; &lt;&lt; std::endl;
        for (const auto&amp; obj : detectedObjects) {
            std::cout &lt;&lt; obj &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = &quot;http://localhost:8080/object-detection&quot;;
        String imagePath = &quot;./demo.jpg&quot;;
        String outputImagePath = &quot;./out.jpg&quot;;

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode detectedObjects = result.get(&quot;detectedObjects&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                System.out.println(&quot;\nDetected objects: &quot; + detectedObjects.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/object-detection&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;image&quot;`
            DetectedObjects []map[string]interface{} `json:&quot;detectedObjects&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, outputImagePath)
    fmt.Println(&quot;\nDetected objects:&quot;)
    for _, obj := range respData.Result.DetectedObjects {
        fmt.Println(obj)
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = &quot;http://localhost:8080/object-detection&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nDetected objects:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;detectedObjects&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/object-detection'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nDetected objects:&quot;);
    console.log(result[&quot;detectedObjects&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/object-detection&quot;;
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nDetected objects:\n&quot;;
print_r($result[&quot;detectedObjects&quot;]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. Refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md) for detailed edge deployment procedures.

Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Object Detection pipeline do not meet your requirements for precision or speed in your specific scenario, you can try to further <b>fine-tune</b> the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the General Object Detection pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Object Detection pipeline includes an object detection module, if the performance of the pipeline does not meet expectations, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/object_detection.md#四二次开发) section in the [Object Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/object_detection.md) and use your private dataset to fine-tune the object detection model.

### 4.2 Model Application
After fine-tuning the model with your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: PicoDet-S  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Then, refer to the command line method or Python script method in the local experience, and load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference of the General Object Detection pipeline, the Python command is:

```bash
paddlex --pipeline object_detection --input general_object_detection_002.png --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline object_detection --input general_object_detection_002.png --device npu:0
```
If you want to use the General Object Detection Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
