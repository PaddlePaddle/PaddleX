---
comments: true
---

# Object Detection Module Development Tutorial

## I. Overview
The object detection module is a crucial component in computer vision systems, responsible for locating and marking regions containing specific objects in images or videos. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The object detection module typically outputs bounding boxes for the target regions, which are then passed as input to the object recognition module for further processing.

## II. List of Supported Models

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PicoDet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>14.68 / 5.81</td>
<td>47.32 / 47.32</td>
<td>20.9 M</td>
<td rowspan="2">PP-PicoDet is a lightweight object detection algorithm for full-size, wide-angle targets, considering the computational capacity of mobile devices. Compared to traditional object detection algorithms, PP-PicoDet has a smaller model size and lower computational complexity, achieving higher speed and lower latency while maintaining detection accuracy.</td>
</tr>
<tr>
<td>PicoDet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Trained Model</a></td>
<td>29.1</td>
<td>7.98 / 2.33</td>
<td>14.82 / 5.60</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Trained Model</a></td>
<td>52.9</td>
<td>33.55 / 10.46</td>
<td>189.05 / 189.05</td>
<td>185.3 M</td>
<td rowspan="2">PP-YOLOE_plus is an upgraded version of the high-precision cloud-edge integrated model PP-YOLOE, developed by Baidu's PaddlePaddle vision team. By using the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the model's end-to-end inference speed.</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Trained Model</a></td>
<td>43.7</td>
<td>12.16 / 4.58</td>
<td>73.86 / 52.90</td>
<td>28.3 M</td>
</tr>
<tr>
<td>RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>56.3</td>
<td>115.92 / 28.16</td>
<td>971.32 / 971.32</td>
<td>435.8 M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detector. The model features an efficient hybrid encoder to meet both model performance and throughput requirements, efficiently handling multi-scale features, and proposes an accelerated and optimized query selection mechanism to optimize the dynamics of decoder queries. RT-DETR supports flexible end-to-end inference speeds by using different decoders.</td>
</tr>
<tr>
<td>RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>53.0</td>
<td>35.00 / 10.45</td>
<td>495.51 / 495.51</td>
<td>113.7 M</td>
</tr>
</table>

&gt; ❗ The above list features the <b>6 core models</b> that the image classification module primarily supports. In total, this module supports <b>37 models</b>. The complete list of models is as follows:

<details><summary> 👉Details of Model List</summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>41.1</td>
<td>135.92 / 135.92</td>
<td>nan / nan</td>
<td>245.4 M</td>
<td rowspan="2">Cascade-FasterRCNN is an improved version of the Faster R-CNN object detection model. By coupling multiple detectors and optimizing detection results using different IoU thresholds, it addresses the mismatch problem between training and prediction stages, enhancing the accuracy of object detection.</td>
</tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Trained Model</a></td>
<td>45.0</td>
<td>138.23 / 138.23</td>
<td>nan / nan</td>
<td>246.2 M</td>
</tr>
<tr>
<td>CenterNet-DLA-34</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-DLA-34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-DLA-34_pretrained.pdparams">Trained Model</a></td>
<td>37.6</td>
<td>nan / nan</td>
<td>nan / nan</td>
<td>75.4 M</td>
<td rowspan="2">CenterNet is an anchor-free object detection model that treats the keypoints of the object to be detected as a single point—the center point of its bounding box, and performs regression through these keypoints.</td>
</tr>
<tr>
<td>CenterNet-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>38.9</td>
<td>nan / nan</td>
<td>nan / nan</td>
<td>319.7 M</td>
</tr>
<tr>
<td>DETR-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DETR-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DETR-R50_pretrained.pdparams">Trained Model</a></td>
<td>42.3</td>
<td>62.91 / 17.33</td>
<td>392.63 / 392.63</td>
<td>159.3 M</td>
<td>DETR is a transformer-based object detection model proposed by Facebook. It achieves end-to-end object detection without the need for predefined anchor boxes or NMS post-processing strategies.</td>
</tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">Trained Model</a></td>
<td>37.8</td>
<td>83.33 / 31.64</td>
<td>nan / nan</td>
<td>137.5 M</td>
<td rowspan="9">Faster R-CNN is a typical two-stage object detection model that first generates region proposals and then performs classification and regression on these proposals. Compared to its predecessors R-CNN and Fast R-CNN, Faster R-CNN's main improvement lies in the region proposal aspect, using a Region Proposal Network (RPN) to provide region proposals instead of traditional selective search. RPN is a Convolutional Neural Network (CNN) that shares convolutional features with the detection network, reducing the computational overhead of region proposals.</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td>
<td>38.4</td>
<td>107.08 / 35.40</td>
<td>nan / nan</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>39.5</td>
<td>109.36 / 36.00</td>
<td>nan / nan</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Trained Model</a></td>
<td>41.4</td>
<td>109.06 / 36.19</td>
<td>nan / nan</td>
<td>148.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>36.7</td>
<td>496.33 / 109.12</td>
<td>nan / nan</td>
<td>120.2 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101-FPN_pretrained.pdparams">Trained Model</a></td>
<td>41.4</td>
<td>148.21 / 42.21</td>
<td>nan / nan</td>
<td>216.3 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101_pretrained.pdparams">Trained Model</a></td>
<td>39.0</td>
<td>538.58 / 120.88</td>
<td>nan / nan</td>
<td>188.1 M</td>
</tr>
<tr>
<td>FasterRCNN-ResNeXt101-vd-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">Trained Model</a></td>
<td>43.4</td>
<td>258.01 / 58.25</td>
<td>nan / nan</td>
<td>360.6 M</td>
</tr>
<tr>
<td>FasterRCNN-Swin-Tiny-FPN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-Swin-Tiny-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-Swin-Tiny-FPN_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>nan / nan</td>
<td>nan / nan</td>
<td>159.8 M</td>
</tr>
<tr>
<td>FCOS-ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>39.6</td>
<td>106.13 / 28.32</td>
<td>721.79 / 721.79</td>
<td>124.2 M</td>
<td>FCOS is an anchor-free object detection model that performs dense predictions. It uses the backbone of RetinaNet and directly regresses the width and height of the target object on the feature map, predicting the object's category and centerness (the degree of offset of pixels on the feature map from the object's center), which is eventually used as a weight to adjust the object score.</td>
</tr>
<tr>
<td>PicoDet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>14.68 / 5.81</td>
<td>47.32 / 47.32</td>
<td>20.9 M</td>
<td rowspan="4">PP-PicoDet is a lightweight object detection algorithm designed for full-size and wide-aspect-ratio targets, with a focus on mobile device computation. Compared to traditional object detection algorithms, PP-PicoDet boasts smaller model sizes and lower computational complexity, achieving higher speeds and lower latency while maintaining detection accuracy.</td>
</tr>
<tr>
<td>PicoDet-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">Trained Model</a></td>
<td>37.5</td>
<td>9.62 / 3.23</td>
<td>23.75 / 14.88</td>
<td>16.8 M</td>
</tr>
<tr>
<td>PicoDet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Trained Model</a></td>
<td>29.1</td>
<td>7.98 / 2.33</td>
<td>14.82 / 5.60</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PicoDet-XS</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">Trained Model</a></td>
<td>26.2</td>
<td>9.66 / 2.75</td>
<td>19.15 / 7.24</td>
<td>5.7 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Trained Model</a></td>
<td>52.9</td>
<td>33.55 / 10.46</td>
<td>189.05 / 189.05</td>
<td>185.3 M</td>
<td rowspan="4">PP-YOLOE_plus is an iteratively optimized and upgraded version of PP-YOLOE, a high-precision cloud-edge integrated model developed by Baidu PaddlePaddle's Vision Team. By leveraging the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the end-to-end inference speed of the model.</td>
</tr>
<tr>
<td>PP-YOLOE_plus-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">Trained Model</a></td>
<td>49.8</td>
<td>19.52 / 7.46</td>
<td>113.36 / 113.36</td>
<td>82.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Trained Model</a></td>
<td>43.7</td>
<td>12.16 / 4.58</td>
<td>73.86 / 52.90</td>
<td>28.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">Trained Model</a></td>
<td>54.7</td>
<td>58.87 / 15.84</td>
<td>292.93 / 292.93</td>
<td>349.4 M</td>
</tr>
<tr>
<td>RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>56.3</td>
<td>115.92 / 28.16</td>
<td>971.32 / 971.32</td>
<td>435.8 M</td>
<td rowspan="5">RT-DETR is the first real-time end-to-end object detector. It features an efficient hybrid encoder that balances model performance and throughput, efficiently processes multi-scale features, and introduces an accelerated and optimized query selection mechanism to dynamize decoder queries. RT-DETR supports flexible end-to-end inference speeds through the use of different decoders.</td>
</tr>
<tr>
<td>RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>53.0</td>
<td>35.00 / 10.45</td>
<td>495.51 / 495.51</td>
<td>113.7 M</td>
</tr>
<tr>
<td>RT-DETR-R18</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">Trained Model</a></td>
<td>46.5</td>
<td>20.21 / 6.23</td>
<td>266.01 / 266.01</td>
<td>70.7 M</td>
</tr>
<tr>
<td>RT-DETR-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R50_pretrained.pdparams">Trained Model</a></td>
<td>53.1</td>
<td>42.14 / 11.31</td>
<td>523.97 / 523.97</td>
<td>149.1 M</td>
</tr>
<tr>
<td>RT-DETR-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-X_pretrained.pdparams">Trained Model</a></td>
<td>54.8</td>
<td>61.24 / 15.83</td>
<td>647.08 / 647.08</td>
<td>232.9 M</td>
</tr>
<tr>
<td>YOLOv3-DarkNet53</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">Trained Model</a></td>
<td>39.1</td>
<td>41.58 / 10.10</td>
<td>158.78 / 158.78</td>
<td>219.7 M</td>
<td rowspan="3">YOLOv3 is a real-time end-to-end object detector that utilizes a unique single Convolutional Neural Network (CNN) to frame the object detection problem as a regression task, enabling real-time detection. The model employs multi-scale detection to enhance performance across different object sizes.</td>
</tr>
<tr>
<td>YOLOv3-MobileNetV3</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-MobileNetV3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-MobileNetV3_pretrained.pdparams">Trained Model</a></td>
<td>31.4</td>
<td>16.53 / 5.70</td>
<td>60.44 / 60.44</td>
<td>83.8 M</td>
</tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-ResNet50_vd_DCN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-ResNet50_vd_DCN_pretrained.pdparams">Trained Model</a></td>
<td>40.6</td>
<td>32.91 / 10.07</td>
<td>225.72 / 224.32</td>
<td>163.0 M</td>
</tr>
<tr>
<td>YOLOX-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-L_pretrained.pdparams">Trained Model</a></td>
<td>50.1</td>
<td>121.19 / 13.55</td>
<td>295.38 / 274.15</td>
<td>192.5 M</td>
<td rowspan="6">Building upon YOLOv3's framework, YOLOX significantly boosts detection performance in complex scenarios by incorporating Decoupled Head, Data Augmentation, Anchor Free, and SimOTA components.</td>
</tr>
<tr>
<td>YOLOX-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-M_pretrained.pdparams">Trained Model</a></td>
<td>46.9</td>
<td>87.19 / 10.09</td>
<td>183.95 / 172.67</td>
<td>90.0 M</td>
</tr>
<tr>
<td>YOLOX-N</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">Trained Model</a></td>
<td>26.1</td>
<td>53.31 / 45.02</td>
<td>69.69 / 59.18</td>
<td>3.4 M</td>
</tr>
<tr>
<td>YOLOX-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-S_pretrained.pdparams">Trained Model</a></td>
<td>40.4</td>
<td>129.52 / 13.19</td>
<td>181.39 / 179.01</td>
<td>32.0 M</td>
</tr>
<tr>
<td>YOLOX-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-T_pretrained.pdparams">Trained Model</a></td>
<td>32.9</td>
<td>66.81 / 61.31</td>
<td>92.30 / 83.90</td>
<td>18.1 M</td>
</tr>
<tr>
<td>YOLOX-X</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-X_pretrained.pdparams">Trained Model</a></td>
<td>51.8</td>
<td>156.40 / 20.17</td>
<td>480.14 / 454.35</td>
<td>351.5 M</td>
</tr>
</table>
<p><b>Note: The precision metrics mentioned are based on the <a href="https://cocodataset.org/#home">COCO2017</a> validation set mAP(0.5:0.95). All model GPU inference times are measured on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p></details>

## III. Quick Integration

&gt; ❗ Before proceeding with quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can perform object detection inference with just a few lines of code. You can easily switch between models within the module and integrate the object detection inference into your projects. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png) to your local machine.

```python
from paddlex import create_model
model = create_model("PicoDet-S")
output = model.predict("general_object_detection_002.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

<details><summary>👉 <b>The result obtained after running is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'general_object_detection_002.png', 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188614249229431, 'coordinate': [661.351806640625, 93.0582275390625, 870.759033203125, 305.9371337890625]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7745078206062317, 'coordinate': [76.80911254882812, 274.7490539550781, 330.5422058105469, 520.0427856445312]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7271787524223328, 'coordinate': [285.3264465332031, 94.31749725341797, 469.7364501953125, 297.4034423828125]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5576589703559875, 'coordinate': [310.8041076660156, 361.4362487792969, 685.1868896484375, 712.591552734375]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5490103363990784, 'coordinate': [764.6251831054688, 285.7609558105469, 924.8153076171875, 440.9289245605469]}, {'cls_id': 47, 'label': 'apple', 'score': 0.515821635723114, 'coordinate': [853.9830932617188, 169.4142303466797, 987.802978515625, 303.5861511230469]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.514293372631073, 'coordinate': [0.5308971405029297, 0.32445716857910156, 1072.953369140625, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.510750949382782, 'coordinate': [57.36802673339844, 23.455347061157227, 213.39601135253906, 176.45611572265625]}]}}
```

Parameter meanings are as follows:
- `input_path`: The path of the input image to be predicted.
- `boxes`: Information of the predicted bounding boxes, a list of dictionaries. Each dictionary contains the following information:
  - `cls_id`: Class ID, an integer.
  - `label`: Class label, a string.
  - `score`: Confidence score of the bounding box, a float.
  - `coordinate`: Coordinates of the bounding box, a list [xmin, ymin, xmax, ymax].

</details>

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/obj_det/general_object_detection_002_res.png"/>

**Note:** Due to network issues, the above URL may not be accessible. If you need to access this link, please check the validity of the URL and try again. If the problem persists, it may be related to the link itself or the network connection.

Related methods, parameters, and explanations are as follows:

* `create_model` instantiates an object detection model (here, `PicoDet-S` is used as an example), and the specific explanations are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>Size of the input image; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>int/list</code></td>
<td>
<ul>
<li><b>int</b>, such as 640, indicating that the input image will be resized to 640x640</li>
<li><b>List</b>, such as [640, 512], indicating that the input image will be resized to a width of 640 and a height of 512</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>float</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the object detection model is called for inference prediction. The `predict()` method has parameters `input`, `batch_size`, and `threshold`, which are explained as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
<li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>Dictionary</b>, the <code>key</code> of the dictionary must correspond to the specific task, such as <code>"img"</code> for image classification tasks. The <code>value</code> of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code></li>
<li><b>List</b>, elements of the list must be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer</td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results; if not specified, the <code>threshold</code> parameter specified in <code>create_model</code> will be used. If <code>create_model</code> also does not specify it, the default configuration of the PaddleX official model will be used</td>
<td><code>float</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The prediction results are processed, and the prediction result for each sample is of type `dict`. It supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development

If you seek higher precision from existing models, you can leverage PaddleX's custom development capabilities to develop better object detection models. Before developing object detection models with PaddleX, ensure you have installed the object detection related training plugins. For installation instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation

Before model training, prepare the corresponding dataset for the task module. PaddleX provides a data validation feature for each module, and <b>only datasets that pass validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to the [PaddleX Object Detection Task Module Data Annotation Guide](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Download Demo Data

You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Validate your dataset with a single command:

```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
```

After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.
<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 4,
    "train_samples": 701,
    "train_sample_paths": [
      "check_dataset/demo_img/road839.png",
      "check_dataset/demo_img/road363.png",
      "check_dataset/demo_img/road148.png",
      "check_dataset/demo_img/road237.png",
      "check_dataset/demo_img/road733.png",
      "check_dataset/demo_img/road861.png",
      "check_dataset/demo_img/road762.png",
      "check_dataset/demo_img/road515.png",
      "check_dataset/demo_img/road754.png",
      "check_dataset/demo_img/road173.png"
    ],
    "val_samples": 176,
    "val_sample_paths": [
      "check_dataset/demo_img/road218.png",
      "check_dataset/demo_img/road681.png",
      "check_dataset/demo_img/road138.png",
      "check_dataset/demo_img/road544.png",
      "check_dataset/demo_img/road596.png",
      "check_dataset/demo_img/road857.png",
      "check_dataset/demo_img/road203.png",
      "check_dataset/demo_img/road589.png",
      "check_dataset/demo_img/road655.png",
      "check_dataset/demo_img/road245.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/det_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 4;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 704;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 176;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul>
<p>Additionally, the dataset verification also analyzes the distribution of sample numbers across all classes in the dataset and generates a histogram (histogram.png) for visualization:</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/obj_det/01.png"/></p></details>


### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Object detection supports converting datasets in <code>VOC</code> and <code>LabelMe</code> formats to <code>COCO</code> format.</p>
<p>Parameters related to dataset validation can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion. Object detection supports converting <code>VOC</code> and <code>LabelMe</code> format datasets to <code>COCO</code> format. Default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, the source dataset format needs to be set. Default is <code>null</code>, with optional values <code>VOC</code>, <code>LabelMe</code>, <code>VOCWithUnlabeled</code>, <code>LabelMeWithUnlabeled</code>;
For example, if you want to convert a <code>LabelMe</code> format dataset to <code>COCO</code> format, taking the following <code>LabelMe</code> format dataset as an example, you need to modify the configuration as follows:</li>
</ul>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_labelme_examples.tar -P ./dataset
tar -xf ./dataset/det_labelme_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples
</code></pre>
<p>Of course, the above parameters also support being set by appending command line arguments. Taking a <code>LabelMe</code> format dataset as an example:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When <code>True</code>, dataset splitting is performed. Default is <code>False</code>;</li>
<li><code>train_percent</code>: If the dataset is re-split, the percentage of the training set needs to be set. The type is any integer between 0-100, and it needs to ensure that the sum with <code>val_percent</code> is 100;</li>
<li><code>val_percent</code>: If the dataset is re-split, the percentage of the validation set needs to be set. The type is any integer between 0-100, and it needs to ensure that the sum with <code>train_percent</code> is 100;
For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, you need to modify the configuration file as follows:</li>
</ul>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
</code></pre>
<p>After dataset splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>


### 4.2 Model Training
Model training can be completed with a single command, taking the training of the object detection model PicoDet-S as an example:

```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PicoDet-S.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves the model weight files, with the default being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PicoDet-S.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference

* To perform inference predictions through the command line, use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_object_detection_002.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PicoDet-S.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

1.<b>Pipeline Integration</b>

The object detection module can be integrated into the [General Object Detection Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/object_detection.en.md) of PaddleX. Simply replace the model path to update the object detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the object detection module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.
