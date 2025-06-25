---
comments: true
---

# Image Classification Module Development Tutorial

## I. Overview
The image classification module is a crucial component in computer vision systems, responsible for categorizing input images. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. Typically, the image classification module receives an image as input and, through deep learning or other machine learning algorithms, classifies it into predefined categories based on its characteristics and content. For instance, in an animal recognition system, the image classification module might need to classify an input image as "cat," "dog," "horse," etc. The classification results from the image classification module are then output for use by other modules or systems.

## II. List of Supported Models


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">Training Model</a></td>
<td>85.36</td>
<td>12.03 / 2.49</td>
<td>60.86 / 42.69</td>
<td>331</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_small_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">Training Model</a></td>
<td>68.2</td>
<td>4.23 / 0.78</td>
<td>5.24 / 1.48</td>
<td>10.5</td>
</tr>
<tr>
<td>PP-HGNet_small</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNet_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">Training Model</a></td>
<td>81.51</td>
<td>5.87 / 1.68</td>
<td>25.58 / 18.50</td>
<td>86.5</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">Training Model</a></td>
<td>77.77</td>
<td>4.41 / 0.87</td>
<td>10.58 / 1.87</td>
<td>21.4</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">Training Model</a></td>
<td>83.57</td>
<td>7.05 / 1.16</td>
<td>16.23 / 7.55</td>
<td>70.4</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B6_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">Training Model</a></td>
<td>86.30</td>
<td>13.86 / 3.28</td>
<td>67.25 / 56.70</td>
<td>268.4</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">Training Model</a></td>
<td>71.32</td>
<td>2.59 / 0.68</td>
<td>3.18 / 1.19</td>
<td>10.5</td>
</tr>
<tr>
<td>ResNet50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">Training Model</a></td>
<td>76.5</td>
<td>6.25 / 1.17</td>
<td>15.93 / 9.72</td>
<td>90.8</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_tiny_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">Training Model</a></td>
<td>81.10</td>
<td>7.11 / 2.01</td>
<td>62.72 / 47.35</td>
<td>100.1</td>
</tr>
</table>

> ❗ The above list features the <b>9 core models</b> that the image classification module primarily supports. In total, this module supports <b>80 models</b>. The complete list of models is as follows:

<details><summary> 👉Details of Model List</summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">Training Model</a></td>
<td>85.36</td>
<td>12.03 / 2.49</td>
<td>60.86 / 42.69</td>
<td>331</td>
<td rowspan="2">CLIP is an image classification model based on the correlation between vision and language. It adopts contrastive learning and pre-training methods to achieve unsupervised or weakly supervised image classification, especially suitable for large-scale datasets. By mapping images and texts into the same representation space, the model learns general features, exhibiting good generalization ability and interpretability. With relatively good training errors, it performs well in many downstream tasks.</td>
</tr>
<tr>
<td>CLIP_vit_large_patch14_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_large_patch14_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_large_patch14_224_pretrained.pdparams">Training Model</a></td>
<td>88.1</td>
<td>49.15 / 9.75</td>
<td>223.16 / 206.49</td>
<td>1040</td>
</tr>
<tr>
<td>ConvNeXt_base_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ConvNeXt_base_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_224_pretrained.pdparams">Training Model</a></td>
<td>83.84</td>
<td>11.37 / 5.65</td>
<td>143.98 / 52.31</td>
<td>313.9</td>
<td rowspan="6">The ConvNeXt series of models were proposed by Meta in 2022, based on the CNN architecture. This series of models builds upon ResNet, incorporating the advantages of SwinTransformer, including training strategies and network structure optimization ideas, to improve the pure CNN architecture network. It explores the performance limits of convolutional neural networks. The ConvNeXt series of models possesses many advantages of convolutional neural networks, including high inference efficiency and ease of migration to downstream tasks.</td>
</tr>
<tr>
<td>ConvNeXt_base_384</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ConvNeXt_base_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_384_pretrained.pdparams">Training Model</a></td>
<td>84.90</td>
<td>29.48 / 11.17</td>
<td>293.76 / 134.27</td>
<td>313.9</td>
</tr>
<tr>
<td>ConvNeXt_large_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ConvNeXt_large_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_224_pretrained.pdparams">Training Model</a></td>
<td>84.26</td>
<td>22.99 / 12.73</td>
<td>220.79 / 113.24</td>
<td>700.7</td>
</tr>
<tr>
<td>ConvNeXt_large_384</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ConvNeXt_large_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_384_pretrained.pdparams">Training Model</a></td>
<td>85.27</td>
<td>58.90 / 24.63</td>
<td>509.48 / 260.27</td>
<td>700.7</td>
</tr>
<tr>
<td>ConvNeXt_small</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ConvNeXt_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_small_pretrained.pdparams">Training Model</a></td>
<td>83.13</td>
<td>7.72 / 4.35</td>
<td>95.92 / 33.34</td>
<td>178.0</td>
</tr>
<tr>
<td>ConvNeXt_tiny</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ConvNeXt_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_tiny_pretrained.pdparams">Training Model</a></td>
<td>82.03</td>
<td>6.00 / 2.47</td>
<td>63.59 / 18.23</td>
<td>104.1</td>
</tr>
<tr>
<td>FasterNet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-L_pretrained.pdparams">Training Model</a></td>
<td>83.5</td>
<td>11.96 / 2.68</td>
<td>51.93 / 35.33</td>
<td>357.1</td>
<td rowspan="6">FasterNet is a neural network designed to improve runtime speed. Its key improvements are as follows:<br/>
1. Re-examined popular operators and found that low FLOPS mainly stem from frequent memory accesses, especially in depthwise convolutions;<br/>
2. Proposed Partial Convolution (PConv) to extract image features more efficiently by reducing redundant computations and memory accesses;<br/>
3. Launched the FasterNet series of models based on PConv, a new design scheme that achieves significantly higher runtime speeds on various devices without compromising model task performance.</td>
</tr>
<tr>
<td>FasterNet-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterNet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-M_pretrained.pdparams">Training Model</a></td>
<td>83.0</td>
<td>11.17 / 2.16</td>
<td>38.49 / 21.17</td>
<td>204.6</td>
</tr>
<tr>
<td>FasterNet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-S_pretrained.pdparams">Training Model</a></td>
<td>81.3</td>
<td>7.70 / 1.24</td>
<td>19.51 / 11.22</td>
<td>119.3</td>
</tr>
<tr>
<td>FasterNet-T0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterNet-T0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T0_pretrained.pdparams">Training Model</a></td>
<td>71.9</td>
<td>4.73 / 0.82</td>
<td>6.40 / 1.96</td>
<td>15.1</td>
</tr>
<tr>
<td>FasterNet-T1</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterNet-T1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T1_pretrained.pdparams">Training Model</a></td>
<td>75.9</td>
<td>4.80 / 0.80</td>
<td>8.14 / 3.13</td>
<td>29.2</td>
</tr>
<tr>
<td>FasterNet-T2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterNet-T2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T2_pretrained.pdparams">Training Model</a></td>
<td>79.1</td>
<td>6.10 / 0.88</td>
<td>12.71 / 5.35</td>
<td>57.4</td>
</tr>
<tr>
<td>MobileNetV1_x0_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV1_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_5_pretrained.pdparams">Training Model</a></td>
<td>63.5</td>
<td>1.98 / 0.51</td>
<td>2.50 / 1.04</td>
<td>4.8</td>
<td rowspan="4">MobileNetV1 is a network released by Google in 2017 for mobile devices or embedded devices. This network decomposes traditional convolution operations into depthwise separable convolutions, which are a combination of Depthwise convolution and Pointwise convolution. Compared to traditional convolutional networks, this combination can significantly reduce the number of parameters and computations. Additionally, this network can be used for image classification and other vision tasks.</td>
</tr>
<tr>
<td>MobileNetV1_x0_25</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV1_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_25_pretrained.pdparams">Training Model</a></td>
<td>51.4</td>
<td>1.99 / 0.45</td>
<td>1.82 / 0.73</td>
<td>1.8</td>
</tr>
<tr>
<td>MobileNetV1_x0_75</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV1_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_75_pretrained.pdparams">Training Model</a></td>
<td>68.8</td>
<td>2.33 / 0.41</td>
<td>3.33 / 1.34</td>
<td>9.3</td>
</tr>
<tr>
<td>MobileNetV1_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV1_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x1_0_pretrained.pdparams">Training Model</a></td>
<td>71.0</td>
<td>2.31 / 0.45</td>
<td>3.91 / 1.89</td>
<td>15.2</td>
</tr>
<tr>
<td>MobileNetV2_x0_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV2_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_5_pretrained.pdparams">Training Model</a></td>
<td>65.0</td>
<td>3.58 / 0.62</td>
<td>3.86 / 1.23</td>
<td>7.1</td>
<td rowspan="5">MobileNetV2 is a lightweight network proposed by Google following MobileNetV1. Compared to MobileNetV1, MobileNetV2 introduces Linear bottlenecks and Inverted residual blocks as the basic structure of the network. By stacking these basic modules extensively, the network structure of MobileNetV2 is formed. Finally, it achieves higher classification accuracy with only half the FLOPs of MobileNetV1.</td>
</tr>
<tr>
<td>MobileNetV2_x0_25</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV2_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_25_pretrained.pdparams">Training Model</a></td>
<td>53.2</td>
<td>3.05 / 0.66</td>
<td>3.30 / 0.98</td>
<td>5.5</td>
</tr>
<tr>
<td>MobileNetV2_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV2_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_0_pretrained.pdparams">Training Model</a></td>
<td>72.2</td>
<td>3.85 / 0.63</td>
<td>5.50 / 1.87</td>
<td>12.6</td>
</tr>
<tr>
<td>MobileNetV2_x1_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV2_x1_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_5_pretrained.pdparams">Training Model</a></td>
<td>74.1</td>
<td>3.93 / 0.73</td>
<td>8.84 / 3.12</td>
<td>25.0</td>
</tr>
<tr>
<td>MobileNetV2_x2_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV2_x2_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x2_0_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>3.89 / 0.79</td>
<td>10.36 / 4.50</td>
<td>41.2</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_large_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_5_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>4.60 / 0.77</td>
<td>5.32 / 1.58</td>
<td>9.6</td>
<td rowspan="10">MobileNetV3 is a NAS-based lightweight network proposed by Google in 2019. To further enhance performance, relu and sigmoid activation functions are replaced with hard_swish and hard_sigmoid activation functions, respectively. Additionally, some improvement strategies specifically designed to reduce network computations are introduced.</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_35</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_large_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_35_pretrained.pdparams">Training Model</a></td>
<td>64.3</td>
<td>4.44 / 0.75</td>
<td>5.20 / 1.50</td>
<td>7.5</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_75</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_large_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_75_pretrained.pdparams">Training Model</a></td>
<td>73.1</td>
<td>5.30 / 0.85</td>
<td>6.02 / 1.93</td>
<td>14.0</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_large_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_0_pretrained.pdparams">Training Model</a></td>
<td>75.3</td>
<td>5.38 / 0.81</td>
<td>7.16 / 2.19</td>
<td>19.5</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_25</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_large_x1_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_25_pretrained.pdparams">Training Model</a></td>
<td>76.4</td>
<td>5.54 / 0.84</td>
<td>7.06 / 2.84</td>
<td>26.5</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_small_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_5_pretrained.pdparams">Training Model</a></td>
<td>59.2</td>
<td>3.87 / 0.77</td>
<td>4.90 / 1.32</td>
<td>6.8</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_35</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_small_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_35_pretrained.pdparams">Training Model</a></td>
<td>53.0</td>
<td>3.68 / 0.77</td>
<td>3.94 / 1.27</td>
<td>6.0</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_75</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_small_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_75_pretrained.pdparams">Training Model</a></td>
<td>66.0</td>
<td>3.92 / 0.77</td>
<td>4.68 / 1.39</td>
<td>8.5</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_small_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">Training Model</a></td>
<td>68.2</td>
<td>4.23 / 0.78</td>
<td>5.24 / 1.48</td>
<td>10.5</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_25</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV3_small_x1_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_25_pretrained.pdparams">Training Model</a></td>
<td>70.7</td>
<td>4.59 / 0.79</td>
<td>5.36 / 1.63</td>
<td>13.0</td>
</tr>
<tr>
<td>MobileNetV4_conv_large</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV4_conv_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_large_pretrained.pdparams">Training Model</a></td>
<td>83.4</td>
<td>9.04 / 2.28</td>
<td>34.34 / 22.01</td>
<td>125.2</td>
<td rowspan="5">MobileNetV4 is an efficient architecture specifically designed for mobile devices. Its core lies in the introduction of the UIB (Universal Inverted Bottleneck) module, a unified and flexible structure that integrates IB (Inverted Bottleneck), ConvNeXt, FFN (Feed Forward Network), and the latest ExtraDW (Extra Depthwise) module. Alongside UIB, Mobile MQA, a customized attention block for mobile accelerators, was also introduced, achieving up to 39% significant acceleration. Furthermore, MobileNetV4 introduces a novel Neural Architecture Search (NAS) scheme to enhance the effectiveness of the search process.</td>
</tr>
<tr>
<td>MobileNetV4_conv_medium</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV4_conv_medium_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_medium_pretrained.pdparams">Training Model</a></td>
<td>79.9</td>
<td>5.70 / 1.05</td>
<td>13.78 / 5.64</td>
<td>37.6</td>
</tr>
<tr>
<td>MobileNetV4_conv_small</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV4_conv_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_small_pretrained.pdparams">Training Model</a></td>
<td>74.6</td>
<td>3.81 / 0.55</td>
<td>5.24 / 1.50</td>
<td>14.7</td>
</tr>
<tr>
<td>MobileNetV4_hybrid_large</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV4_hybrid_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_large_pretrained.pdparams">Training Model</a></td>
<td>83.8</td>
<td>13.43 / 4.28</td>
<td>61.16 / 31.06</td>
<td>145.1</td>
</tr>
<tr>
<td>MobileNetV4_hybrid_medium</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileNetV4_hybrid_medium_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_medium_pretrained.pdparams">Training Model</a></td>
<td>80.5</td>
<td>11.82 / 1.30</td>
<td>22.01 / 6.06</td>
<td>42.9</td>
</tr>
<tr>
<td>PP-HGNet_base</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNet_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_base_pretrained.pdparams">Training Model</a></td>
<td>85.0</td>
<td>13.43 / 3.81</td>
<td>71.24 / 51.48</td>
<td>249.4</td>
<td rowspan="3">PP-HGNet (High Performance GPU Net) is a high-performance backbone network developed by Baidu PaddlePaddle's vision team, tailored for GPU platforms. This network combines the fundamentals of VOVNet with learnable downsampling layers (LDS Layer), incorporating the advantages of models such as ResNet_vd and PPHGNet. On GPU platforms, this model achieves higher accuracy compared to other SOTA models at the same speed. Specifically, it outperforms ResNet34-0 by 3.8 percentage points and ResNet50-0 by 2.4 percentage points. Under the same SLSD conditions, it ultimately surpasses ResNet50-D by 4.7 percentage points. Additionally, at the same level of accuracy, its inference speed significantly exceeds that of mainstream Vision Transformers.</td>
</tr>
<tr>
<td>PP-HGNet_small</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNet_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">Training Model</a></td>
<td>81.51</td>
<td>5.87 / 1.68</td>
<td>25.58 / 18.50</td>
<td>86.5</td>
</tr>
<tr>
<td>PP-HGNet_tiny</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNet_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_tiny_pretrained.pdparams">Training Model</a></td>
<td>79.83</td>
<td>5.84 / 1.38</td>
<td>17.03 / 10.58</td>
<td>52.4</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">Training Model</a></td>
<td>77.77</td>
<td>4.41 / 0.87</td>
<td>10.58 / 1.87</td>
<td>21.4</td>
<td rowspan="7">PP-HGNetV2 (High Performance GPU Network V2) is the next-generation version of Baidu PaddlePaddle's PP-HGNet, featuring further optimizations and improvements upon its predecessor. It pushes the limits of NVIDIA's "Accuracy-Latency Balance," significantly outperforming other models with similar inference speeds in terms of accuracy. It demonstrates strong performance across various label classification and evaluation scenarios.</td>
</tr>
<tr>
<td>PP-HGNetV2-B1</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B1_pretrained.pdparams">Training Model</a></td>
<td>79.18</td>
<td>4.52 / 0.73</td>
<td>11.98 / 2.28</td>
<td>22.6</td>
</tr>
<tr>
<td>PP-HGNetV2-B2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B2_pretrained.pdparams">Training Model</a></td>
<td>81.74</td>
<td>6.67 / 0.96</td>
<td>14.22 / 4.04</td>
<td>39.9</td>
</tr>
<tr>
<td>PP-HGNetV2-B3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B3_pretrained.pdparams">Training Model</a></td>
<td>82.98</td>
<td>7.47 / 1.94</td>
<td>17.73 / 5.63</td>
<td>57.9</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">Training Model</a></td>
<td>83.57</td>
<td>7.05 / 1.16</td>
<td>16.23 / 7.55</td>
<td>70.4</td>
</tr>
<tr>
<td>PP-HGNetV2-B5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B5_pretrained.pdparams">Training Model</a></td>
<td>84.75</td>
<td>10.38 / 1.95</td>
<td>31.53 / 18.02</td>
<td>140.8</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B6_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">Training Model</a></td>
<td>86.30</td>
<td>13.86 / 3.28</td>
<td>67.25 / 56.70</td>
<td>268.4</td>
</tr>
<tr>
<td>PP-LCNet_x0_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_5_pretrained.pdparams">Training Model</a></td>
<td>63.14</td>
<td>2.41 / 0.60</td>
<td>2.54 / 0.90</td>
<td>6.7</td>
<td rowspan="8">PP-LCNet is a lightweight backbone network developed by Baidu PaddlePaddle's vision team. It enhances model performance without increasing inference time, significantly surpassing other lightweight SOTA models.</td>
</tr>
<tr>
<td>PP-LCNet_x0_25</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_pretrained.pdparams">Training Model</a></td>
<td>51.86</td>
<td>2.16 / 0.60</td>
<td>2.73 / 0.77</td>
<td>5.5</td>
</tr>
<tr>
<td>PP-LCNet_x0_35</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_35_pretrained.pdparams">Training Model</a></td>
<td>58.09</td>
<td>2.18 / 0.60</td>
<td>2.32 / 0.89</td>
<td>5.9</td>
</tr>
<tr>
<td>PP-LCNet_x0_75</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_75_pretrained.pdparams">Training Model</a></td>
<td>68.18</td>
<td>2.61 / 0.58</td>
<td>3.00 / 1.09</td>
<td>8.4</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">Training Model</a></td>
<td>71.32</td>
<td>2.59 / 0.68</td>
<td>3.18 / 1.19</td>
<td>10.5</td>
</tr>
<tr>
<td>PP-LCNet_x1_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_5_pretrained.pdparams">Training Model</a></td>
<td>73.71</td>
<td>2.60 / 0.68</td>
<td>3.98 / 1.66</td>
<td>16.0</td>
</tr>
<tr>
<td>PP-LCNet_x2_0</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x2_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_0_pretrained.pdparams">Training Model</a></td>
<td>75.18</td>
<td>2.53 / 0.68</td>
<td>5.21 / 2.24</td>
<td>23.2</td>
</tr>
<tr>
<td>PP-LCNet_x2_5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x2_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_5_pretrained.pdparams">Training Model</a></td>
<td>76.60</td>
<td>2.76 / 0.67</td>
<td>6.78 / 3.20</td>
<td>32.1</td>
</tr>
<tr>
<tr>
<td>PP-LCNetV2_base</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNetV2_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_base_pretrained.pdparams">Training Model</a></td>
<td>77.05</td>
<td>4.04 / 0.62</td>
<td>6.80 / 2.67</td>
<td>23.7</td>
<td rowspan="3">The PP-LCNetV2 image classification model is the next-generation version of PP-LCNet, self-developed by Baidu PaddlePaddle's vision team. Based on PP-LCNet, it has undergone further optimization and improvements, primarily utilizing re-parameterization strategies to combine depthwise convolutions with varying kernel sizes and optimizing pointwise convolutions, Shortcuts, etc. Without using additional data, the PPLCNetV2_base model achieves over 77% Top-1 Accuracy on the ImageNet dataset for image classification, while maintaining an inference time of less than 4.4 ms on Intel CPU platforms.</td>
</tr>
<tr>
<td>PP-LCNetV2_large</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNetV2_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_large_pretrained.pdparams">Training Model</a></td>
<td>78.51</td>
<td>4.91 / 0.85</td>
<td>10.30 / 5.38</td>
<td>37.3</td>
</tr>
<tr>
<td>PP-LCNetV2_small</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNetV2_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_small_pretrained.pdparams">Training Model</a></td>
<td>73.97</td>
<td>3.07 / 0.60</td>
<td>4.28 / 1.58</td>
<td>14.6</td>
</tr>
<tr>
<tr>
<td>ResNet18_vd</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet18_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_vd_pretrained.pdparams">Training Model</a></td>
<td>72.3</td>
<td>2.87 / 0.77</td>
<td>7.91 / 4.64</td>
<td>41.5</td>
<td rowspan="11">The ResNet series of models were introduced in 2015, winning the ILSVRC2015 competition with a top-5 error rate of 3.57%. This network innovatively proposed residual structures, which are stacked to construct the ResNet network. Experiments have shown that using residual blocks can effectively improve convergence speed and accuracy.</td>
</tr>
<tr>
<td>ResNet18</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">Training Model</a></td>
<td>71.0</td>
<td>2.63 / 0.74</td>
<td>6.30 / 4.16</td>
<td>41.5</td>
</tr>
<tr>
<td>ResNet34_vd</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet34_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_vd_pretrained.pdparams">Training Model</a></td>
<td>76.0</td>
<td>4.47 / 1.09</td>
<td>14.30 / 8.33</td>
<td>77.3</td>
</tr>
<tr>
<td>ResNet34</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">Training Model</a></td>
<td>74.6</td>
<td>4.20 / 1.07</td>
<td>12.53 / 7.83</td>
<td>77.3</td>
</tr>
<tr>
<td>ResNet50_vd</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet50_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_vd_pretrained.pdparams">Training Model</a></td>
<td>79.1</td>
<td>6.66 / 1.23</td>
<td>16.34 / 10.00</td>
<td>90.8</td>
</tr>
<tr>
<td>ResNet50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">Training Model</a></td>
<td>76.5</td>
<td>6.25 / 1.17</td>
<td>15.93 / 9.72</td>
<td>90.8</td>
</tr>
<tr>
<td>ResNet101_vd</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet101_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_vd_pretrained.pdparams">Training Model</a></td>
<td>80.2</td>
<td>11.93 / 2.07</td>
<td>32.47 / 23.62</td>
<td>158.4</td>
</tr>
<tr>
<td>ResNet101</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">Training Model</a></td>
<td>77.6</td>
<td>13.73 / 2.06</td>
<td>29.69 / 17.72</td>
<td>158.4</td>
</tr>
<tr>
<td>ResNet152_vd</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet152_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_vd_pretrained.pdparams">Training Model</a></td>
<td>80.6</td>
<td>20.70 / 2.82</td>
<td>43.90 / 27.91</td>
<td>214.3</td>
</tr>
<tr>
<td>ResNet152</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet152_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">Training Model</a></td>
<td>78.3</td>
<td>17.86 / 2.79</td>
<td>46.19 / 26.00</td>
<td>214.2</td>
</tr>
<tr>
<td>ResNet200_vd</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet200_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet200_vd_pretrained.pdparams">Training Model</a></td>
<td>80.9</td>
<td>22.55 / 3.54</td>
<td>58.54 / 35.70</td>
<td>266.0</td>
</tr>
<tr>
<tr>
<td>StarNet-S1</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/StarNet-S1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S1_pretrained.pdparams">Training Model</a></td>
<td>73.6</td>
<td>6.24 / 0.96</td>
<td>8.78 / 2.44</td>
<td>11.2</td>
<td rowspan="4">StarNet focuses on exploring the untapped potential of "star operations" (i.e., element-wise multiplication) in network design. It reveals that star operations can map inputs to high-dimensional, nonlinear feature spaces, a process akin to kernel tricks but without the need to expand the network size. Consequently, StarNet, a simple yet powerful prototype network, is further proposed, demonstrating exceptional performance and low latency under compact network structures and limited computational resources.</td>
</tr>
<tr>
<td>StarNet-S2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/StarNet-S2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S2_pretrained.pdparams">Training Model</a></td>
<td>74.8</td>
<td>4.78 / 0.85</td>
<td>7.24 / 2.48</td>
<td>14.3</td>
</tr>
<tr>
<td>StarNet-S3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/StarNet-S3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S3_pretrained.pdparams">Training Model</a></td>
<td>77.0</td>
<td>6.77 / 1.07</td>
<td>9.69 / 3.35</td>
<td>22.2</td>
</tr>
<tr>
<td>StarNet-S4</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/StarNet-S4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S4_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>9.01 / 1.48</td>
<td>14.79 / 4.58</td>
<td>28.9</td>
</tr>
<tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_base_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window7_224_pretrained.pdparams">Training Model</a></td>
<td>83.37</td>
<td>13.04 / 10.77</td>
<td>133.79 / 118.45</td>
<td>340</td>
<td rowspan="6">SwinTransformer is a novel vision Transformer network that can serve as a general-purpose backbone for computer vision tasks. SwinTransformer consists of a hierarchical Transformer structure represented by shifted windows. Shifted windows restrict self-attention computations to non-overlapping local windows while allowing cross-window connections, thereby enhancing network performance.</td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window12_384</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_base_patch4_window12_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window12_384_pretrained.pdparams">Training Model</a></td>
<td>84.17</td>
<td>33.99 / 28.42</td>
<td>400.19 / 317.36</td>
<td>311.4</td>
</tr>
<tr>
<td>SwinTransformer_large_patch4_window7_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_large_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window7_224_pretrained.pdparams">Training Model</a></td>
<td>86.19</td>
<td>23.69 / 6.18</td>
<td>198.60 / 177.18</td>
<td>694.8</td>
</tr>
<tr>
<td>SwinTransformer_large_patch4_window12_384</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_large_patch4_window12_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window12_384_pretrained.pdparams">Training Model</a></td>
<td>87.06</td>
<td>68.07 / 14.84</td>
<td>609.07 / 525.72</td>
<td>696.1</td>
</tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_small_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_small_patch4_window7_224_pretrained.pdparams">Training Model</a></td>
<td>83.21</td>
<td>12.17 / 3.51</td>
<td>111.03 / 92.51</td>
<td>175.6</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SwinTransformer_tiny_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">Training Model</a></td>
<td>81.10</td>
<td>7.11 / 2.01</td>
<td>62.72 / 47.35</td>
<td>100.1</td>
</tr>
</tr></tr></tr></tr></table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                <li><strong>Test Dataset：</strong><a href="https://www.image-net.org/index.php">ImageNet-1k</a> validation set.</li>
              <li><strong>Hardware Configuration：</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environments: Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>Inference Mode Description</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>Mode</th>
            <th>GPU Configuration </th>
            <th>CPU Configuration </th>
            <th>Acceleration Technology Combination</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Normal Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of pre-selected precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

</details>

## <span id="lable">III. Quick Integration</span>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can complete image classification module inference with just a few lines of code. You can switch between models in this module freely, and you can also integrate the model inference of the image classification module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0")
output = model.predict("general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```josn
{'res': {'input_path': 'test_imgs/general_image_classification_001.jpg', 'class_ids': [296, 279, 270, 537, 356], 'scores': [0.7915499806404114, 0.0173799991607666, 0.014279999770224094, 0.013009999878704548, 0.01221999991685152], 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Arctic fox, white fox, Alopex lagopus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'dogsled, dog sled, dog sleigh', 'weasel']}}
```

The meanings of the running results parameters are as follows:
- `input_path`: Indicates the path of the input image.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF is currently being processed; otherwise, it is `None`.
- `class_ids`: Indicates the class IDs of the prediction results.
- `scores`: Indicates the confidence scores of the prediction results.
- `label_names`: Indicates the class names of the prediction results.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/image_classification/general_image_classification_001_res.jpg"/>

**Note:** Due to network issues, the above URL may not be accessible. If you need to access this link, please check the validity of the URL and try again. If the problem persists, it may be related to the link itself or the network connection.

Related methods, parameters, and explanations are as follows:

* `create_model` instantiates an image classification model (here, `PP-LCNet_x1_0` is used as an example), and the specific explanations are as follows:
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
<td><code>PP-LCNet_x1_0</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for model inference</td>
<td><code>str</code></td>
<td>It supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the image classification model is called for inference prediction. The `predict()` method has parameters `input` and `batch_size`, which are explained as follows:

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
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">Example</a></li>
<li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>，elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
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

For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better image classification models. Before using PaddleX to develop image classification models, please ensure that you have installed the relevant model training plugins for image classification in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and <b>only data that passes data validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use your own private dataset for subsequent model training, please refer to the [PaddleX Image Classification Task Module Data Annotation Guide](../../../data_annotations/cv_modules/image_classification.en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to a specified folder:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
One command is all you need to complete data validation:

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Validation Results Details (Click to Expand)</b></summary>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "dataset/label.txt",
    "num_classes": 102,
    "train_samples": 1020,
    "train_sample_paths": [
      "check_dataset/demo_img/image_01904.jpg",
      "check_dataset/demo_img/image_06940.jpg"
    ],
    "val_samples": 1020,
    "val_sample_paths": [
      "check_dataset/demo_img/image_01937.jpg",
      "check_dataset/demo_img/image_06958.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "cls_flowers_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
</code></pre>
<p>The above validation results, with check_pass being True, indicate that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 102;</li>
<li><code>attributes.train_samples</code>: The number of training set samples in this dataset is 1020;</li>
<li><code>attributes.val_samples</code>: The number of validation set samples in this dataset is 1020;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visual samples in the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visual samples in the validation set of this dataset;</li>
</ul>
<p>Additionally, the dataset validation analyzes the sample number distribution across all classes in the dataset and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/image_classification/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Image classification does not currently support data conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. The following are example explanations for some of the parameters in the configuration file:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When set to <code>True</code>, the dataset format will be converted. The default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, you need to set the percentage of the training set, which should be an integer between 0-100, ensuring that the sum with <code>val_percent</code> equals 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, you need to modify the configuration file as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>These parameters also support being set through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
A single command can complete the model training. Taking the training of the image classification model PP-LCNet_x1_0 as an example:
```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```

the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0.yaml`. When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path of the training dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file parameter instructions for the corresponding task module of the model [PaddleX Common Model Configuration File Parameters](../../instructions/config_parameters_common.en.md).


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
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.json</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
<li>Notice: Since Paddle 3.0.0, the format of storing static graph network structure has changed to json(the current<code>.json</code> file) from protobuf(the former<code>.pdmodel</code> file) to be compatible with PIR and more flexible and scalable.</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. Using PaddleX for model evaluation, a single command can complete the model evaluation:
```bash
python main.py -c  paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`
* Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).
* New Feature: Paddle 3.0 support CINN (Compiler Infrastructure for Neural Networks) to accelerate training speed when using GPU device. Please specify `-o Train.dy2st=True` to enable it.

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed successfully and the model's evaluation metrics, including val.top1, val.top5;</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the Training Model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg) to your local machine.

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_image_classification_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-LCNet_x1_0.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

1.<b>Pipeline Integration</b>

The image classification module can be integrated into the [General Image Classification Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/image_classification.en.md) of PaddleX. Simply replace the model path to update the image classification module of the relevant pipeline. In pipeline integration, you can use high-performance inference and serving deployment to deploy your obtained model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the image classification module. You can refer to the Python example code in <a href="#lable">Quick Integration</a>  and simply replace the model with the path to your Training Model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).
