---
comments: true
---

# General Image Classification Pipeline Tutorial

## 1. Introduction to the General Image Classification Pipeline
Image classification is a technique that assigns images to predefined categories. It is widely applied in object recognition, scene understanding, and automatic annotation. Image classification can identify various objects such as animals, plants, traffic signs, and categorize them based on their features. By leveraging deep learning models, image classification can automatically extract image features and perform accurate classification.This pipeline also offers a flexible service-oriented deployment approach, supporting the use of multiple programming languages on various hardware platforms. Moreover, this pipeline provides the capability for secondary development. You can train and optimize models on your own dataset based on this pipeline, and the trained models can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/01.png"/>
<b>The General Image Classification Pipeline includes an image classification module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">Trained Model</a></td>
<td>85.36</td>
<td>12.84 / 2.82</td>
<td>60.52 / 60.52</td>
<td>306.5 M</td>
<td>The general high-precision image classification model of the large visual model CLIP fine-tuned on the ImageNet1k dataset</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>68.2</td>
<td>3.76 / 0.53</td>
<td>5.11 / 1.43</td>
<td>10.5 M</td>
<td>MobileNetV3 is a new lightweight network based on NAS proposed by Google in 2019. To further improve performance, the relu and sigmoid activation functions are replaced with hard_swish and hard_sigmoid activation functions, respectively. Additionally, several strategies specifically aimed at reducing the computational load of the network have been introduced.</td>
</tr>
<tr>
<td>PP-HGNet_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">Trained Model</a></td>
<td>81.51</td>
<td>5.12 / 1.73</td>
<td>25.01 / 25.01</td>
<td>86.5 M</td>
<td>PP-HGNet (High Performance GPU Net) is a high-performance backbone network developed by the Baidu PaddlePaddle Vision Team, specifically optimized for GPU platforms. This network builds upon VOVNet and incorporates a learnable downsampling layer (LDS Layer), integrating the advantages of models such as ResNet_vd and PPHGNet. The model achieves higher accuracy compared to other state-of-the-art (SOTA) models at the same speed on GPU platforms.</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">Trained Model</a></td>
<td>77.77</td>
<td>3.83 / 0.57</td>
<td>9.95 / 2.37</td>
<td>21.4 M</td>
<td rowspan="3">PP-HGNetV2 (High Performance GPU Network V2) is the next-generation version of PP-HGNet developed by the Baidu PaddlePaddle Vision Team. Building upon PP-HGNet, it has been further optimized and improved. Ultimately, on NVIDIA GPU devices, it achieves an extreme "Accuracy-Latency Balance," with accuracy significantly surpassing other models of the same inference speed.</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">Trained Model</a></td>
<td>83.57</td>
<td>5.47 / 1.10</td>
<td>14.42 / 9.89</td>
<td>70.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">Trained Model</a></td>
<td>86.30</td>
<td>12.25 / 3.76</td>
<td>62.29 / 62.29</td>
<td>268.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>71.32</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>10.5 M</td>
<td>PP-LCNet_x1_0 is designed with a specific backbone network for Intel CPU devices and their acceleration library MKLDNN. Compared to other lightweight state-of-the-art (SOTA) models, this backbone network can further enhance model performance without increasing inference time, ultimately significantly surpassing existing SOTA models.</td>
</tr>
<tr>
<td>ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>76.5</td>
<td>6.44 / 1.16</td>
<td>15.04 / 11.63</td>
<td>90.8 M</td>
<td>The ResNet series of models was proposed in 2015 and won the championship in the ILSVRC2015 competition with a top-5 error rate of 3.57%. The network innovatively introduced the residual structure, and by stacking multiple residual structures, the ResNet network was constructed.</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">Trained Model</a></td>
<td>81.10</td>
<td>6.66 / 2.15</td>
<td>60.45 / 60.45</td>
<td>100.1 M</td>
<td>SwinTransformer is a new type of visual Transformer network that can be used as a general-purpose backbone network in the field of computer vision. SwinTransformer consists of a hierarchical Transformer structure represented by shifted windows. The shifted windows confine the self-attention computation to non-overlapping local windows while allowing cross-window connections, thereby enhancing the network's performance.</td>
</tr>
</table>

> ❗ The above list features the <b>9 core models</b> that the image classification module primarily supports. In total, this module supports <b>80 models</b>. The complete list of models is as follows:

<details><summary> 👉Details of Model List</summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Accuracy (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">Trained Model</a></td>
<td>85.36</td>
<td>12.84 / 2.82</td>
<td>60.52 / 60.52</td>
<td>306.5 M</td>
<td rowspan="2">CLIP is an image classification model based on the correlation between vision and language. It adopts contrastive learning and pre-training methods to achieve unsupervised or weakly supervised image classification, especially suitable for large-scale datasets. By mapping images and texts into the same representation space, the model learns general features, exhibiting good generalization ability and interpretability. With relatively good training errors, it performs well in many downstream tasks.</td>
</tr>
<tr>
<td>CLIP_vit_large_patch14_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_large_patch14_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_large_patch14_224_pretrained.pdparams">Trained Model</a></td>
<td>88.1</td>
<td>51.72 / 11.13</td>
<td>238.07 / 238.07</td>
<td>1.04 G</td>
</tr>
<tr>
<td>ConvNeXt_base_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_224_pretrained.pdparams">Trained Model</a></td>
<td>83.84</td>
<td>13.18 / 12.14</td>
<td>128.39 / 81.78</td>
<td>313.9 M</td>
<td rowspan="6">The ConvNeXt series of models were proposed by Meta in 2022, based on the CNN architecture. This series of models builds upon ResNet, incorporating the advantages of SwinTransformer, including training strategies and network structure optimization ideas, to improve the pure CNN architecture network. It explores the performance limits of convolutional neural networks. The ConvNeXt series of models possesses many advantages of convolutional neural networks, including high inference efficiency and ease of migration to downstream tasks.</td>
</tr>
<tr>
<td>ConvNeXt_base_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_384_pretrained.pdparams">Trained Model</a></td>
<td>84.90</td>
<td>32.15 / 30.52</td>
<td>279.36 / 220.35</td>
<td>313.9 M</td>
</tr>
<tr>
<td>ConvNeXt_large_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_224_pretrained.pdparams">Trained Model</a></td>
<td>84.26</td>
<td>26.51 / 7.21</td>
<td>213.32 / 157.22</td>
<td>700.7 M</td>
</tr>
<tr>
<td>ConvNeXt_large_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_384_pretrained.pdparams">Trained Model</a></td>
<td>85.27</td>
<td>67.07 / 65.26</td>
<td>494.04 / 438.97</td>
<td>700.7 M</td>
</tr>
<tr>
<td>ConvNeXt_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_small_pretrained.pdparams">Trained Model</a></td>
<td>83.13</td>
<td>9.05 / 8.21</td>
<td>97.94 / 55.29</td>
<td>178.0 M</td>
</tr>
<tr>
<td>ConvNeXt_tiny</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_tiny_pretrained.pdparams">Trained Model</a></td>
<td>82.03</td>
<td>5.12 / 2.06</td>
<td>63.96 / 29.77</td>
<td>104.1 M</td>
</tr>
<tr>
<td>FasterNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-L_pretrained.pdparams">Trained Model</a></td>
<td>83.5</td>
<td>15.67 / 3.10</td>
<td>52.24 / 52.24</td>
<td>357.1 M</td>
<td rowspan="6">FasterNet is a neural network designed to improve runtime speed. Its key improvements are as follows:<br/>
1. Re-examined popular operators and found that low FLOPS mainly stem from frequent memory accesses, especially in depthwise convolutions;<br/>
2. Proposed Partial Convolution (PConv) to extract image features more efficiently by reducing redundant computations and memory accesses;<br/>
3. Launched the FasterNet series of models based on PConv, a new design scheme that achieves significantly higher runtime speeds on various devices without compromising model task performance.</td>
</tr>
<tr>
<td>FasterNet-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-M_pretrained.pdparams">Trained Model</a></td>
<td>83.0</td>
<td>9.72 / 2.30</td>
<td>35.29 / 35.29</td>
<td>204.6 M</td>
</tr>
<tr>
<td>FasterNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-S_pretrained.pdparams">Trained Model</a></td>
<td>81.3</td>
<td>5.46 / 1.27</td>
<td>20.46 / 18.03</td>
<td>119.3 M</td>
</tr>
<tr>
<td>FasterNet-T0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T0_pretrained.pdparams">Trained Model</a></td>
<td>71.9</td>
<td>4.18 / 0.60</td>
<td>6.34 / 3.44</td>
<td>15.1 M</td>
</tr>
<tr>
<td>FasterNet-T1</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T1_pretrained.pdparams">Trained Model</a></td>
<td>75.9</td>
<td>4.24 / 0.64</td>
<td>9.57 / 5.20</td>
<td>29.2 M</td>
</tr>
<tr>
<td>FasterNet-T2</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T2_pretrained.pdparams">Trained Model</a></td>
<td>79.1</td>
<td>3.87 / 0.78</td>
<td>11.14 / 9.98</td>
<td>57.4 M</td>
</tr>
<tr>
<td>MobileNetV1_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_5_pretrained.pdparams">Trained Model</a></td>
<td>63.5</td>
<td>1.39 / 0.28</td>
<td>2.74 / 1.02</td>
<td>4.8 M</td>
<td rowspan="4">MobileNetV1 is a network released by Google in 2017 for mobile devices or embedded devices. This network decomposes traditional convolution operations into depthwise separable convolutions, which are a combination of Depthwise convolution and Pointwise convolution. Compared to traditional convolutional networks, this combination can significantly reduce the number of parameters and computations. Additionally, this network can be used for image classification and other vision tasks.</td>
</tr>
<tr>
<td>MobileNetV1_x0_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_25_pretrained.pdparams">Trained Model</a></td>
<td>51.4</td>
<td>1.32 / 0.30</td>
<td>2.04 / 0.58</td>
<td>1.8 M</td>
</tr>
<tr>
<td>MobileNetV1_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_75_pretrained.pdparams">Trained Model</a></td>
<td>68.8</td>
<td>1.75 / 0.33</td>
<td>3.41 / 1.57</td>
<td>9.3 M</td>
</tr>
<tr>
<td>MobileNetV1_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>71.0</td>
<td>1.89 / 0.34</td>
<td>4.01 / 2.17</td>
<td>15.2 M</td>
</tr>
<tr>
<td>MobileNetV2_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_5_pretrained.pdparams">Trained Model</a></td>
<td>65.0</td>
<td>3.17 / 0.48</td>
<td>4.52 / 1.35</td>
<td>7.1 M</td>
<td rowspan="5">MobileNetV2 is a lightweight network proposed by Google following MobileNetV1. Compared to MobileNetV1, MobileNetV2 introduces Linear bottlenecks and Inverted residual blocks as the basic structure of the network. By stacking these basic modules extensively, the network structure of MobileNetV2 is formed. Finally, it achieves higher classification accuracy with only half the FLOPs of MobileNetV1.</td>
</tr>
<tr>
<td>MobileNetV2_x0_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_25_pretrained.pdparams">Trained Model</a></td>
<td>53.2</td>
<td>2.80 / 0.46</td>
<td>3.92 / 0.98</td>
<td>5.5 M</td>
</tr>
<tr>
<td>MobileNetV2_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>72.2</td>
<td>3.57 / 0.49</td>
<td>5.63 / 2.51</td>
<td>12.6 M</td>
</tr>
<tr>
<td>MobileNetV2_x1_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_5_pretrained.pdparams">Trained Model</a></td>
<td>74.1</td>
<td>3.58 / 0.62</td>
<td>8.02 / 4.49</td>
<td>25.0 M</td>
</tr>
<tr>
<td>MobileNetV2_x2_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x2_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x2_0_pretrained.pdparams">Trained Model</a></td>
<td>75.2</td>
<td>3.56 / 0.74</td>
<td>10.24 / 6.83</td>
<td>41.2 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_5_pretrained.pdparams">Trained Model</a></td>
<td>69.2</td>
<td>3.79 / 0.62</td>
<td>6.76 / 1.61</td>
<td>9.6 M</td>
<td rowspan="10">MobileNetV3 is a NAS-based lightweight network proposed by Google in 2019. To further enhance performance, relu and sigmoid activation functions are replaced with hard_swish and hard_sigmoid activation functions, respectively. Additionally, some improvement strategies specifically designed to reduce network computations are introduced.</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_35</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_35_pretrained.pdparams">Trained Model</a></td>
<td>64.3</td>
<td>3.70 / 0.60</td>
<td>5.54 / 1.41</td>
<td>7.5 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_75_pretrained.pdparams">Trained Model</a></td>
<td>73.1</td>
<td>4.82 / 0.66</td>
<td>7.45 / 2.00</td>
<td>14.0 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>75.3</td>
<td>4.86 / 0.68</td>
<td>6.88 / 2.61</td>
<td>19.5 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_25_pretrained.pdparams">Trained Model</a></td>
<td>76.4</td>
<td>5.08 / 0.71</td>
<td>7.37 / 3.58</td>
<td>26.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_5_pretrained.pdparams">Trained Model</a></td>
<td>59.2</td>
<td>3.41 / 0.57</td>
<td>5.60 / 1.14</td>
<td>6.8 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_35</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_35_pretrained.pdparams">Trained Model</a></td>
<td>53.0</td>
<td>3.49 / 0.60</td>
<td>4.63 / 1.07</td>
<td>6.0 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_75_pretrained.pdparams">Trained Model</a></td>
<td>66.0</td>
<td>3.49 / 0.60</td>
<td>5.19 / 1.28</td>
<td>8.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>68.2</td>
<td>3.76 / 0.53</td>
<td>5.11 / 1.43</td>
<td>10.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_25_pretrained.pdparams">Trained Model</a></td>
<td>70.7</td>
<td>4.23 / 0.58</td>
<td>6.48 / 1.68</td>
<td>13.0 M</td>
</tr>
<tr>
<td>MobileNetV4_conv_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_large_pretrained.pdparams">Trained Model</a></td>
<td>83.4</td>
<td>8.33 / 2.24</td>
<td>33.56 / 23.70</td>
<td>125.2 M</td>
<td rowspan="5">MobileNetV4 is an efficient architecture specifically designed for mobile devices. Its core lies in the introduction of the UIB (Universal Inverted Bottleneck) module, a unified and flexible structure that integrates IB (Inverted Bottleneck), ConvNeXt, FFN (Feed Forward Network), and the latest ExtraDW (Extra Depthwise) module. Alongside UIB, Mobile MQA, a customized attention block for mobile accelerators, was also introduced, achieving up to 39% significant acceleration. Furthermore, MobileNetV4 introduces a novel Neural Architecture Search (NAS) scheme to enhance the effectiveness of the search process.</td>
</tr>
<tr>
<td>MobileNetV4_conv_medium</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_medium_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_medium_pretrained.pdparams">Trained Model</a></td>
<td>79.9</td>
<td>6.81 / 0.92</td>
<td>12.47 / 6.27</td>
<td>37.6 M</td>
</tr>
<tr>
<td>MobileNetV4_conv_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_small_pretrained.pdparams">Trained Model</a></td>
<td>74.6</td>
<td>3.25 / 0.46</td>
<td>4.42 / 1.54</td>
<td>14.7 M</td>
</tr>
<tr>
<td>MobileNetV4_hybrid_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_large_pretrained.pdparams">Trained Model</a></td>
<td>83.8</td>
<td>12.27 / 4.18</td>
<td>58.64 / 58.64</td>
<td>145.1 M</td>
</tr>
<tr>
<td>MobileNetV4_hybrid_medium</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_medium_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_medium_pretrained.pdparams">Trained Model</a></td>
<td>80.5</td>
<td>12.08 / 1.34</td>
<td>24.69 / 8.10</td>
<td>42.9 M</td>
</tr>
<tr>
<td>PP-HGNet_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_base_pretrained.pdparams">Trained Model</a></td>
<td>85.0</td>
<td>14.10 / 4.19</td>
<td>68.92 / 68.92</td>
<td>249.4 M</td>
<td rowspan="3">PP-HGNet (High Performance GPU Net) is a high-performance backbone network developed by Baidu PaddlePaddle's vision team, tailored for GPU platforms. This network combines the fundamentals of VOVNet with learnable downsampling layers (LDS Layer), incorporating the advantages of models such as ResNet_vd and PPHGNet. On GPU platforms, this model achieves higher accuracy compared to other SOTA models at the same speed. Specifically, it outperforms ResNet34-0 by 3.8 percentage points and ResNet50-0 by 2.4 percentage points. Under the same SLSD conditions, it ultimately surpasses ResNet50-D by 4.7 percentage points. Additionally, at the same level of accuracy, its inference speed significantly exceeds that of mainstream Vision Transformers.</td>
</tr>
<tr>
<td>PP-HGNet_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">Trained Model</a></td>
<td>81.51</td>
<td>5.12 / 1.73</td>
<td>25.01 / 25.01</td>
<td>86.5 M</td>
</tr>
<tr>
<td>PP-HGNet_tiny</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_tiny_pretrained.pdparams">Trained Model</a></td>
<td>79.83</td>
<td>3.28 / 1.29</td>
<td>16.40 / 15.97</td>
<td>52.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">Trained Model</a></td>
<td>77.77</td>
<td>3.83 / 0.57</td>
<td>9.95 / 2.37</td>
<td>21.4 M</td>
<td rowspan="7">PP-HGNetV2 (High Performance GPU Network V2) is the next-generation version of Baidu PaddlePaddle's PP-HGNet, featuring further optimizations and improvements upon its predecessor. It pushes the limits of NVIDIA's "Accuracy-Latency Balance," significantly outperforming other models with similar inference speeds in terms of accuracy. It demonstrates strong performance across various label classification and evaluation scenarios.</td>
</tr>
<tr>
<td>PP-HGNetV2-B1</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B1_pretrained.pdparams">Trained Model</a></td>
<td>79.18</td>
<td>3.87 / 0.62</td>
<td>8.77 / 3.79</td>
<td>22.6 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B2</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B2_pretrained.pdparams">Trained Model</a></td>
<td>81.74</td>
<td>5.73 / 0.86</td>
<td>15.11 / 7.05</td>
<td>39.9 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B3</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B3_pretrained.pdparams">Trained Model</a></td>
<td>82.98</td>
<td>6.26 / 1.01</td>
<td>18.47 / 10.34</td>
<td>57.9 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">Trained Model</a></td>
<td>83.57</td>
<td>5.47 / 1.10</td>
<td>14.42 / 9.89</td>
<td>70.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B5_pretrained.pdparams">Trained Model</a></td>
<td>84.75</td>
<td>10.24 / 1.96</td>
<td>29.71 / 29.71</td>
<td>140.8 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">Trained Model</a></td>
<td>86.30</td>
<td>12.25 / 3.76</td>
<td>62.29 / 62.29</td>
<td>268.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_5_pretrained.pdparams">Trained Model</a></td>
<td>63.14</td>
<td>2.28 / 0.42</td>
<td>2.86 / 0.83</td>
<td>6.7 M</td>
<td rowspan="8">PP-LCNet is a lightweight backbone network developed by Baidu PaddlePaddle's vision team. It enhances model performance without increasing inference time, significantly surpassing other lightweight SOTA models.</td>
</tr>
<tr>
<td>PP-LCNet_x0_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_pretrained.pdparams">Trained Model</a></td>
<td>51.86</td>
<td>1.89 / 0.45</td>
<td>2.49 / 0.68</td>
<td>5.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_35</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_35_pretrained.pdparams">Trained Model</a></td>
<td>58.09</td>
<td>1.94 / 0.41</td>
<td>2.73 / 0.77</td>
<td>5.9 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_75_pretrained.pdparams">Trained Model</a></td>
<td>68.18</td>
<td>2.30 / 0.41</td>
<td>2.95 / 1.07</td>
<td>8.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">Trained Model</a></td>
<td>71.32</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>10.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_5_pretrained.pdparams">Trained Model</a></td>
<td>73.71</td>
<td>2.33 / 0.53</td>
<td>4.17 / 2.29</td>
<td>16.0 M</td>
</tr>
<tr>
<td>PP-LCNet_x2_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_0_pretrained.pdparams">Trained Model</a></td>
<td>75.18</td>
<td>2.40 / 0.51</td>
<td>5.37 / 3.46</td>
<td>23.2 M</td>
</tr>
<tr>
<td>PP-LCNet_x2_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_5_pretrained.pdparams">Trained Model</a></td>
<td>76.60</td>
<td>2.36 / 0.61</td>
<td>6.29 / 5.05</td>
<td>32.1 M</td>
</tr>
<tr>
<tr>
<td>PP-LCNetV2_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_base_pretrained.pdparams">Trained Model</a></td>
<td>77.05</td>
<td>3.33 / 0.55</td>
<td>6.86 / 3.77</td>
<td>23.7 M</td>
<td rowspan="3">The PP-LCNetV2 image classification model is the next-generation version of PP-LCNet, self-developed by Baidu PaddlePaddle's vision team. Based on PP-LCNet, it has undergone further optimization and improvements, primarily utilizing re-parameterization strategies to combine depthwise convolutions with varying kernel sizes and optimizing pointwise convolutions, Shortcuts, etc. Without using additional data, the PPLCNetV2_base model achieves over 77% Top-1 Accuracy on the ImageNet dataset for image classification, while maintaining an inference time of less than 4.4 ms on Intel CPU platforms.</td>
</tr>
<tr>
<td>PP-LCNetV2_large </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_large_pretrained.pdparams">Trained Model</a></td>
<td>78.51</td>
<td>4.37 / 0.71</td>
<td>9.43 / 8.07</td>
<td>37.3 M</td>
</tr>
<tr>
<td>PP-LCNetV2_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_small_pretrained.pdparams">Trained Model</a></td>
<td>73.97</td>
<td>2.53 / 0.41</td>
<td>5.14 / 1.98</td>
<td>14.6 M</td>
</tr>
<tr>
<tr>
<td>ResNet18_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_vd_pretrained.pdparams">Trained Model</a></td>
<td>72.3</td>
<td>2.47 / 0.61</td>
<td>6.97 / 5.15</td>
<td>41.5 M</td>
<td rowspan="11">The ResNet series of models were introduced in 2015, winning the ILSVRC2015 competition with a top-5 error rate of 3.57%. This network innovatively proposed residual structures, which are stacked to construct the ResNet network. Experiments have shown that using residual blocks can effectively improve convergence speed and accuracy.</td>
</tr>
<tr>
<td>ResNet18 </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">Trained Model</a></td>
<td>71.0</td>
<td>2.35 / 0.67</td>
<td>6.35 / 4.61</td>
<td>41.5 M</td>
</tr>
<tr>
<td>ResNet34_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_vd_pretrained.pdparams">Trained Model</a></td>
<td>76.0</td>
<td>4.01 / 1.03</td>
<td>11.99 / 9.86</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet34</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">Trained Model</a></td>
<td>74.6</td>
<td>3.99 / 1.02</td>
<td>12.42 / 9.81</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet50_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_vd_pretrained.pdparams">Trained Model</a></td>
<td>79.1</td>
<td>6.04 / 1.16</td>
<td>16.08 / 12.07</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">Trained Model</a></td>
<td>76.5</td>
<td>6.44 / 1.16</td>
<td>15.04 / 11.63</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet101_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_vd_pretrained.pdparams">Trained Model</a></td>
<td>80.2</td>
<td>11.16 / 2.07</td>
<td>32.14 / 32.14</td>
<td>158.4 M</td>
</tr>
<tr>
<td>ResNet101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">Trained Model</a></td>
<td>77.6</td>
<td>10.91 / 2.06</td>
<td>31.14 / 22.93</td>
<td>158.4 M</td>
</tr>
<tr>
<td>ResNet152_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_vd_pretrained.pdparams">Trained Model</a></td>
<td>80.6</td>
<td>15.96 / 2.99</td>
<td>49.33 / 49.33</td>
<td>214.3 M</td>
</tr>
<tr>
<td>ResNet152</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">Trained Model</a></td>
<td>78.3</td>
<td>15.61 / 2.90</td>
<td>47.33 / 36.60</td>
<td>214.2 M</td>
</tr>
<tr>
<td>ResNet200_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet200_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet200_vd_pretrained.pdparams">Trained Model</a></td>
<td>80.9</td>
<td>24.20 / 3.69</td>
<td>62.62 / 62.62</td>
<td>266.0 M</td>
</tr>
<tr>
<tr>
<td>StarNet-S1</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S1_pretrained.pdparams">Trained Model</a></td>
<td>73.6</td>
<td>6.33 / 1.98</td>
<td>7.56 / 3.26</td>
<td>11.2 M</td>
<td rowspan="4">StarNet focuses on exploring the untapped potential of "star operations" (i.e., element-wise multiplication) in network design. It reveals that star operations can map inputs to high-dimensional, nonlinear feature spaces, a process akin to kernel tricks but without the need to expand the network size. Consequently, StarNet, a simple yet powerful prototype network, is further proposed, demonstrating exceptional performance and low latency under compact network structures and limited computational resources.</td>
</tr>
<tr>
<td>StarNet-S2 </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S2_pretrained.pdparams">Trained Model</a></td>
<td>74.8</td>
<td>4.49 / 1.55</td>
<td>7.38 / 3.38</td>
<td>14.3 M</td>
</tr>
<tr>
<td>StarNet-S3</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S3_pretrained.pdparams">Trained Model</a></td>
<td>77.0</td>
<td>6.70 / 1.62</td>
<td>11.05 / 4.76</td>
<td>22.2 M</td>
</tr>
<tr>
<td>StarNet-S4</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S4_pretrained.pdparams">Trained Model</a></td>
<td>79.0</td>
<td>8.50 / 2.86</td>
<td>15.40 / 6.76</td>
<td>28.9 M</td>
</tr>
<tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window7_224_pretrained.pdparams">Trained Model</a></td>
<td>83.37</td>
<td>14.29 / 5.13</td>
<td>130.89 / 130.89</td>
<td>310.5 M</td>
<td rowspan="6">SwinTransformer is a novel vision Transformer network that can serve as a general-purpose backbone for computer vision tasks. SwinTransformer consists of a hierarchical Transformer structure represented by shifted windows. Shifted windows restrict self-attention computations to non-overlapping local windows while allowing cross-window connections, thereby enhancing network performance.</td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window12_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window12_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window12_384_pretrained.pdparams">Trained Model</a></td>
<td>84.17</td>
<td>37.74 / 10.10</td>
<td>362.56 / 362.56</td>
<td>311.4 M</td>
</tr>
<tr>
<td>SwinTransformer_large_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window7_224_pretrained.pdparams">Trained Model</a></td>
<td>86.19</td>
<td>26.48 / 7.94</td>
<td>228.23 / 228.23</td>
<td>694.8 M</td>
</tr>
<tr>
<td>SwinTransformer_large_patch4_window12_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window12_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window12_384_pretrained.pdparams">Trained Model</a></td>
<td>87.06</td>
<td>74.72 / 18.16</td>
<td>652.04 / 652.04</td>
<td>696.1 M</td>
</tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_small_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_small_patch4_window7_224_pretrained.pdparams">Trained Model</a></td>
<td>83.21</td>
<td>10.37 / 3.90</td>
<td>94.20 / 94.20</td>
<td>175.6 M</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">Trained Model</a></td>
<td>81.10</td>
<td>6.66 / 2.15</td>
<td>60.45 / 60.45</td>
<td>100.1 M</td>
</tr>
</tr></tr></tr></tr></table>
<p><b>Note: The above accuracy metrics refer to Top-1 Accuracy on the <a href="https://www.image-net.org/index.php">ImageNet-1k</a> validation set. </b><b>All model GPU inference times are based on NVIDIA Tesla T4 machines, with precision type FP32. CPU inference speeds are based on Intel® Xeon® Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.</b></p></details>

## 2. Quick Start
All model pipelines provided by PaddleX can be quickly experienced. You can experience the general image classification pipeline online in the Star River Community, or you can use the command line or Python locally to experience the effects of the general image classification pipeline.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/100061/webUI) the effects of the general image classification pipeline using the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/02.png"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the results, you can **fine-tune the models in the pipeline using your private data**. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code Platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
Before using the general image classification pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
You can quickly experience the image classification pipeline with a single command. Use [the test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg) and replace `--input` with your local path for prediction.


```bash
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device gpu:0 --save_path ./output/
```

The relevant parameter descriptions can be found in the parameter explanation section of [2.2.2 Python Script Integration](#222-integration-via-python-script).

```bash
{'res': {'input_path': 'general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([296, 170, 356, 258, 248], dtype=int32), 'scores': array([0.62736, 0.03752, 0.03256, 0.0323 , 0.03194], dtype=float32), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}}
```

For the explanation of the running result parameters, you can refer to the result interpretation in [Section 2.2.2 Integration via Python Script](#222-integration-via-python-script).

The visualization results are saved under `save_path`, and the visualization result for image classification is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png"/>

**Note:** Due to network issues, the above URL may not be successfully parsed. If you need the content of this web page, please check the validity of the link and try again. If you do not need the content of this link, you can proceed with the integration as described.

#### 2.2.2 Integration via Python Script
* The above command line is for quick experience and viewing of results. Generally, in projects, integration through code is often required. You can complete the pipeline's fast inference with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_classification")

output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img(save_path="./output/") ## 保存结果可视化图像
    res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
```

In the above Python script, the following steps are executed:

(1) The general image classification pipeline object is instantiated via `create_pipeline()`. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of a pipeline, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying the specific card number of GPUs, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPUs, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the image classification pipeline object is called to perform inference prediction. This method returns a `generator`. Below are the parameters and their descriptions for the `predict()` method:

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
<tbody>
<tr>
<td><code>input</code></td>
<td>The data to be predicted. It supports multiple input types and is required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path of the image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of the image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code>.</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference.</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Use CPU for inference, such as <code>cpu</code>.</li>
<li><b>GPU</b>: Use the specified GPU for inference, such as <code>gpu:0</code> for the first GPU.</li>
<li><b>NPU</b>: Use the specified NPU for inference, such as <code>npu:0</code> for the first NPU.</li>
<li><b>XPU</b>: Use the specified XPU for inference, such as <code>xpu:0</code> for the first XPU.</li>
<li><b>MLU</b>: Use the specified MLU for inference, such as <code>mlu:0</code> for the first MLU.</li>
<li><b>DCU</b>: Use the specified DCU for inference, such as <code>dcu:0</code> for the first DCU.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used. During initialization, it will prioritize the local GPU device 0; if unavailable, it will use the CPU.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>topk</code></td>
<td>The top <code>topk</code> values of the prediction results. If not specified, the default configuration of the official PaddleX model will be used.</td>
<td><code>int</code></td>
<td>
<ul>
<li><b>int</b>, such as 5, which means printing (returning) the top <code>5</code> classes and their corresponding classification probabilities in the prediction results.</li>
</ul>
</td>
<td>5</td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is provided, the saved file name matches the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
</table>

For the explanation of the running result parameters, you can refer to the result interpretation in [Section 2.2.2 Integration via Python Script](#222-integration-via-python-script).

The visualization results are saved under `save_path`, and the visualization result for image classification is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png"/>

#### 2.2.2 Integration via Python Script
* The above command line is for quick experience and viewing of results. Generally, in projects, integration through code is often required. You can complete the pipeline's fast inference with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_classification")

output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img(save_path="./output/") ## 保存结果可视化图像
    res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
```
In the above Python script, the following steps are executed:

(1) The general image classification pipeline object is instantiated via `create_pipeline()`. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of a pipeline, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying the specific card number of GPUs, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPUs, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the image classification pipeline object is called to perform inference prediction. This method returns a `generator`. Below are the parameters and their descriptions for the `predict()` method:

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
<tbody>
<tr>
<td><code>input</code></td>
<td>The data to be predicted. It supports multiple input types and is required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path of the image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of the image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code>.</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference.</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Use CPU for inference, such as <code>cpu</code>.</li>
<li><b>GPU</b>: Use the specified GPU for inference, such as <code>gpu:0</code> for the first GPU.</li>
<li><b>NPU</b>: Use the specified NPU for inference, such as <code>npu:0</code> for the first NPU.</li>
<li><b>XPU</b>: Use the specified XPU for inference, such as <code>xpu:0</code> for the first XPU.</li>
<li><b>MLU</b>: Use the specified MLU for inference, such as <code>mlu:0</code> for the first MLU.</li>
<li><b>DCU</b>: Use the specified DCU for inference, such as <code>dcu:0</code> for the first DCU.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used. During initialization, it will prioritize the local GPU device 0; if unavailable, it will use the CPU.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>topk</code></td>
<td>The top <code>topk</code> values of the prediction results. If not specified, the default configuration of the official PaddleX model will be used.</td>
<td><code>int</code></td>
<td>
<ul>
<li><b>int</b>, such as 5, which means printing (returning) the top <code>5</code> classes and their corresponding classification probabilities in the prediction results.</li>
</ul>
</td>
<td>5</td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is provided, the saved file name will match the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal, with the following explanations for the printed content:

    - `input_path`: `(str)` The input path of the image to be predicted.
    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.
    - `class_ids`: `(List[numpy.ndarray])` The class IDs of the prediction results.
    - `scores`: `(List[numpy.ndarray])` The confidence scores of the prediction results.
    - `label_names`: `(List[str])` The names of the predicted classes.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, the results will be saved directly to that file. Since JSON files do not support saving NumPy arrays, `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualized results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If a file is specified, the results will be saved directly to that file. (It is not recommended to specify a specific file path directly, as multiple images will be overwritten, leaving only the last one.)

* Additionally, you can access the visualized image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format.</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format.</td>
</tr>
</table>

- The prediction results obtained through the `json` attribute are of type `dict`, with content consistent with what is saved by calling the `save_to_json()` method.
- The prediction results returned by the `img` attribute are of type `dict`. The key `res` corresponds to an `Image.Image` object: a visualized image for displaying classification results.

Additionally, you can obtain the configuration file for the image classification pipeline and load it for prediction. You can run the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config image_classification --save_path ./my_path
```
If you have obtained the configuration file, you can customize the settings for the image classification pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/image_classification.yaml")

output = pipeline.predict(
    input="./general_image_classification_001.jpg",
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are pipeline initialization parameters. If you wish to change the initialization parameters for the general image classification pipeline, you can directly modify the parameters in the configuration file and load it for prediction. Additionally, CLI prediction also supports passing a configuration file, simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline directly into your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python-script-method).

In addition, PaddleX also provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have strict performance requirements (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX offers a high-performance inference plugin that deeply optimizes model inference and pre/post-processing to significantly accelerate the end-to-end process. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in practical production environments. By encapsulating inference functionality as a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
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
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Classify the image.</p>
<p><code>POST /image-classification</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
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
<td>The URL of the image file accessible by the server or the Base64-encoded content of the image file.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>topk</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>See the <code>topk</code> parameter description in the <code>predict</code> method documentation.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>categories</code></td>
<td><code>array</code></td>
<td>Image category information.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>| <code>null</code></td>
<td>The image classification result. The image is in JPEG format and is encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>categories</code> is an <code>object</code> with the following properties:</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>Category ID.</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>Category name.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Category score.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
"categories": [
{
"id": 5,
"name": "rabbit",
"score": 0.93
}
],
"image": "xxxxxx"
}
</code></pre></details>
<details><summary>Multi-language Service Call Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/image-classification"  # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode a local image using Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nCategories:")
print(result["categories"])
</code></pre></details>
<details><summary>C++</summary>
<pre><code class="language-cpp">#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode a local image using Base64
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; "Error reading file." &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*="">(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post("/image-classification", headers, body, "application/json");
    // Process the response data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char=""> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; "Output image saved at " &lt;&lt; outputImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; "Unable to open file for writing: " &lt;&lt; outputImagePath &lt;&lt; std::endl;
        }

        auto categories = result["categories"];
        std::cout &lt;&lt; "\nCategories:" &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; "Failed to send HTTP request." &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</char*></unsigned></const></char></iostream></code></pre></details>
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
        String API_URL = "http://localhost:8080/image-classification"; // Service URL
        String imagePath = "./demo.jpg"; // Local image
        String outputImagePath = "./out.jpg"; // Output image

        // Encode the local image using Base64
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64-encoded file content or image URL

        // Create an OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call the API and process the returned data
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                String base64Image = result.get("image").asText();
                JsonNode categories = result.get("categories");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nCategories: " + categories.toString());
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
</code></pre></details>
<details><summary>Go</summary>
<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/image-classification"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    // Encode the local image using Base64
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64-encoded file content or image URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // Call the API
    client := &amp;http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer res.Body.Close()

    // Process the response data
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }

    type Response struct {
        Result struct {
            Image      string   `json:"image"`
            Categories []map[string]interface{} `json:"categories"`
        } `json:"result"`
    }

    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding base64 image data:", err)
        return
    }

    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }

    fmt.Printf("Image saved at %s\n", outputImagePath)
    fmt.Println("\nCategories:")
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
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
    static readonly string API_URL = "http://localhost:8080/image-classification";
    static readonly string imagePath = "./demo.jpg";
    static readonly string outputImagePath = "./out.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode a local image using Base64
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64-encoded file content or image URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the response data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image saved at {outputImagePath}");
        Console.WriteLine("\nCategories:");
        Console.WriteLine(jsonResponse["result"]["categories"].ToString());
    }
}
</code></pre></details>
<details><summary>Node.js</summary>
<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/image-classification';
const imagePath = './demo.jpg';
const outputImagePath = './out.jpg';

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64-encoded file content or image URL
  })
};

// Encode the local image using Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) =&gt; {
    // Process the returned data
    const result = response.data['result'];
    const imageBuffer = Buffer.from(result['image'], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nCategories:");
    console.log(result['categories']);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>
<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/image-classification"; // Service URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// Encode the local image using Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" =&gt; $image_data); // Base64-encoded file content or image URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the response data
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nCategories:\n";
print_r($result["categories"]);
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing capabilities directly on the user's device, allowing the device to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method according to your needs to integrate the model pipeline into subsequent AI applications.

## 4. Secondary Development
If the default model weights provided by the general image classification pipeline are not satisfactory in terms of accuracy or speed in your scenario, you can try to <b>fine-tune</b> the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general image classification pipeline in your scenario.

### 4.1 Model Fine-Tuning

Since the general image classification pipeline includes an image classification module, if the pipeline's performance does not meet expectations, you need to refer to the fine-tuning tutorial links in the table below for model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Fine-Tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate multi-label classification</td>
<td>Multi-label classification module</td>
<td><a href="../../../module_usage/tutorials/cv_modules/image_classification.en.md">Link</a></td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After you complete fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding position in the pipeline configuration file.

```yaml
SubModules:
  ImageClassification:
    module_name: image_classification
    model_name: PP-LCNet_x0_5
    model_dir: null # Replace with the path to the fine-tuned image classification model weights
    batch_size: 4
    topk: 5
```

Subsequently, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you are using Ascend NPU for inference in the general image classification pipeline, the Python command you would use is:

```bash
paddlex --pipeline image_classification \
        --input general_image_classification_001.jpg \
        --save_path ./output \
        --device npu:0
```

If you want to use the general image classification pipeline on a wider variety of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
