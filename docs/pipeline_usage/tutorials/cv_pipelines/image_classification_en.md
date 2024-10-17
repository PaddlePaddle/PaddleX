[简体中文](image_classification.md) | English

# General Image Classification Pipeline Usage Tutorial

## 1. Introduction to the General Image Classification Pipeline
Image classification is a technique that assigns images to predefined categories. It is widely applied in object recognition, scene understanding, and automatic annotation. Image classification can identify various objects such as animals, plants, traffic signs, and categorize them based on their features. By leveraging deep learning models, image classification can automatically extract image features and perform accurate classification.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/01.png)

**The General Image Classification Pipeline includes an image classification module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.**

<details>
   <summary> 👉Details of Model List</summary>

<table>
  <tr>
    <th>Model</th>
    <th>Top-1 Accuracy (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
<tr>
    <td>CLIP_vit_base_patch16_224</td>
    <td>85.36</td>
    <td>13.1957</td>
    <td>285.493</td>
    <td >306.5 M</td>
    <td rowspan="2">CLIP is an image classification model based on the correlation between vision and language. It adopts contrastive learning and pre-training methods to achieve unsupervised or weakly supervised image classification, especially suitable for large-scale datasets. By mapping images and texts into the same representation space, the model learns general features, exhibiting good generalization ability and interpretability. With relatively good training errors, it performs well in many downstream tasks.</td>
  </tr>
  <tr>
    <td>CLIP_vit_large_patch14_224</td>
    <td>88.1</td>
    <td>51.1284</td>
    <td>1131.28</td>
    <td>1.04 G</td>
  </tr>
  <tr>
    <td>ConvNeXt_base_224</td>
    <td>83.84</td>
    <td>12.8473</td>
    <td>1513.87</td>
    <td>313.9 M</td>
    <td rowspan="6">The ConvNeXt series of models were proposed by Meta in 2022, based on the CNN architecture. This series of models builds upon ResNet, incorporating the advantages of SwinTransformer, including training strategies and network structure optimization ideas, to improve the pure CNN architecture network. It explores the performance limits of convolutional neural networks. The ConvNeXt series of models possesses many advantages of convolutional neural networks, including high inference efficiency and ease of migration to downstream tasks.</td>
  </tr>
  <tr>
    <td>ConvNeXt_base_384</td>
    <td>84.90</td>
    <td>31.7607</td>
    <td>3967.05</td>
    <td>313.9 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_large_224</td>
    <td>84.26</td>
    <td>26.8103</td>
    <td>2463.56</td>
    <td>700.7 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_large_384</td>
    <td>85.27</td>
    <td>66.4058</td>
    <td>6598.92</td>
    <td>700.7 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_small</td>
    <td>83.13</td>
    <td>9.74075</td>
    <td>1127.6</td>
    <td>178.0 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_tiny</td>
    <td>82.03</td>
    <td>5.48923</td>
    <td>672.559</td>
    <td>104.1 M</td>
  </tr>
  <tr>
    <td>FasterNet-L</td>
    <td>83.5</td>
    <td>23.4415</td>
    <td>-</td>
    <td>357.1 M</td>
    <td rowspan="6">FasterNet is a neural network designed to improve runtime speed. Its key improvements are as follows:<br>
      1. Re-examined popular operators and found that low FLOPS mainly stem from frequent memory accesses, especially in depthwise convolutions;<br>
      2. Proposed Partial Convolution (PConv) to extract image features more efficiently by reducing redundant computations and memory accesses;<br>
      3. Launched the FasterNet series of models based on PConv, a new design scheme that achieves significantly higher runtime speeds on various devices without compromising model task performance.</td>
  </tr>
  <tr>
    <td>FasterNet-M</td>
    <td>83.0</td>
    <td>21.8936</td>
    <td>-</td>
    <td>204.6 M</td>
  </tr>
  <tr>
    <td>FasterNet-S</td>
    <td>81.3</td>
    <td>13.0409</td>
    <td>-</td>
    <td>119.3 M</td>
  </tr>
  <tr>
    <td>FasterNet-T0</td>
    <td>71.9</td>
    <td>12.2432</td>
    <td>-</td>
    <td>15.1 M</td>
  </tr>
  <tr>
    <td>FasterNet-T1</td>
    <td>75.9</td>
    <td>11.3562</td>
    <td>-</td>
    <td>29.2 M</td>
  </tr>
  <tr>
    <td>FasterNet-T2</td>
    <td>79.1</td>
    <td>10.703</td>
    <td>-</td>
    <td>57.4 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_5</td>
    <td>63.5</td>
    <td>1.86754</td>
    <td>7.48297</td>
    <td>4.8 M</td>
    <td rowspan="4">MobileNetV1 is a network released by Google in 2017 for mobile devices or embedded devices. This network decomposes traditional convolution operations into depthwise separable convolutions, which are a combination of Depthwise convolution and Pointwise convolution. Compared to traditional convolutional networks, this combination can significantly reduce the number of parameters and computations. Additionally, this network can be used for image classification and other vision tasks.</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_25</td>
    <td>51.4</td>
    <td>1.83478</td>
    <td>4.83674</td>
    <td>1.8 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_75</td>
    <td>68.8</td>
    <td>2.57903</td>
    <td>10.6343</td>
    <td>9.3 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x1_0</td>
    <td>71.0</td>
    <td>2.78781</td>
    <td>13.98</td>
    <td>15.2 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x0_5</td>
    <td>65.0</td>
    <td>4.94234</td>
    <td>11.1629</td>
    <td>7.1 M</td>
    <td rowspan="5">MobileNetV2 is a lightweight network proposed by Google following MobileNetV1. Compared to MobileNetV1, MobileNetV2 introduces Linear bottlenecks and Inverted residual blocks as the basic structure of the network. By stacking these basic modules extensively, the network structure of MobileNetV2 is formed. Finally, it achieves higher classification accuracy with only half the FLOPs of MobileNetV1.</td>
  </tr>
  <tr>
    <td>MobileNetV2_x0_25</td>
    <td>53.2</td>
    <td>4.50856</td>
    <td>9.40991</td>
    <td>5.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x1_0</td>
    <td>72.2</td>
    <td>6.12159</td>
    <td>16.0442</td>
    <td>12.6 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x1_5</td>
    <td>74.1</td>
    <td>6.28385</td>
    <td>22.5129</td>
    <td>25.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x2_0</td>
    <td>75.2</td>
    <td>6.12888</td>
    <td>30.8612</td>
    <td>41.2 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_5</td>
    <td>69.2</td>
    <td>6.31302</td>
    <td>14.5588</td>
    <td>9.6 M</td>
    <td rowspan="10">MobileNetV3 is a NAS-based lightweight network proposed by Google in 2019. To further enhance performance, relu and sigmoid activation functions are replaced with hard_swish and hard_sigmoid activation functions, respectively. Additionally, some improvement strategies specifically designed to reduce network computations are introduced.</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_35</td>
    <td>64.3</td>
    <td>5.76207</td>
    <td>13.9041</td>
    <td>7.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_75</td>
    <td>73.1</td>
    <td>8.41737</td>
    <td>16.9506</td>
    <td>14.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x1_0</td>
    <td>75.3</td>
    <td>8.64112</td>
    <td>19.1614</td>
    <td>19.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x1_25</td>
    <td>76.4</td>
    <td>8.73358</td>
    <td>22.1296</td>
    <td>26.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_5</td>
    <td>59.2</td>
    <td>5.16721</td>
    <td>11.2688</td>
    <td>6.8 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_35</td>
    <td>53.0</td>
    <td>5.22053</td>
    <td>11.0055</td>
    <td>6.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_75</td>
    <td>66.0</td>
    <td>5.39831</td>
    <td>12.8313</td>
    <td>8.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_0</td>
    <td>68.2</td>
    <td>6.00993</td>
    <td>12.9598</td>
    <td>10.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_25</td>
    <td>70.7</td>
    <td>6.9589</td>
    <td>14.3995</td>
    <td>13.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_large</td>
    <td>83.4</td>
    <td>12.5485</td>
    <td>51.6453</td>
    <td>125.2 M</td>
    <td rowspan="5">MobileNetV4 is an efficient architecture specifically designed for mobile devices. Its core lies in the introduction of the UIB (Universal Inverted Bottleneck) module, a unified and flexible structure that integrates IB (Inverted Bottleneck), ConvNeXt, FFN (Feed Forward Network), and the latest ExtraDW (Extra Depthwise) module. Alongside UIB, Mobile MQA, a customized attention block for mobile accelerators, was also introduced, achieving up to 39% significant acceleration. Furthermore, MobileNetV4 introduces a novel Neural Architecture Search (NAS) scheme to enhance the effectiveness of the search process.</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_medium</td>
    <td>79.9</td>
    <td>9.65509</td>
    <td>26.6157</td>
    <td>37.6 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_small</td>
    <td>74.6</td>
    <td>5.24172</td>
    <td>11.0893</td>
    <td>14.7 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_hybrid_large</td>
    <td>83.8</td>
    <td>20.0726</td>
    <td>213.769</td>
    <td>145.1 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_hybrid_medium</td>
    <td>80.5</td>
    <td>19.7543</td>
    <td>62.2624</td>
    <td>42.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNet_base</td>
    <td>85.0</td>
    <td>14.2969</td>
    <td>327.114</td>
    <td>249.4 M</td>
    <td rowspan="3">PP-HGNet (High Performance GPU Net) is a high-performance backbone network developed by Baidu PaddlePaddle's vision team, tailored for GPU platforms. This network combines the fundamentals of VOVNet with learnable downsampling layers (LDS Layer), incorporating the advantages of models such as ResNet_vd and PPHGNet. On GPU platforms, this model achieves higher accuracy compared to other SOTA models at the same speed. Specifically, it outperforms ResNet34-0 by 3.8 percentage points and ResNet50-0 by 2.4 percentage points. Under the same SLSD conditions, it ultimately surpasses ResNet50-D by 4.7 percentage points. Additionally, at the same level of accuracy, its inference speed significantly exceeds that of mainstream Vision Transformers.</td>
  </tr>
  <tr>
    <td>PP-HGNet_small</td>
    <td>81.51</td>
    <td>5.50661</td>
    <td>119.041</td>
    <td>86.5 M</td>
  </tr>
  <tr>
    <td>PP-HGNet_tiny</td>
    <td>79.83</td>
    <td>5.22006</td>
    <td>69.396</td>
    <td>52.4 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B0</td>
    <td>77.77</td>
    <td>6.53694</td>
    <td>23.352</td>
    <td>21.4 M</td>
    <td rowspan="7">PP-HGNetV2 (High Performance GPU Network V2) is the next-generation version of Baidu PaddlePaddle's PP-HGNet, featuring further optimizations and improvements upon its predecessor. It pushes the limits of NVIDIA's "Accuracy-Latency Balance," significantly outperforming other models with similar inference speeds in terms of accuracy. It demonstrates strong performance across various label classification and evaluation scenarios.</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B1</td>
    <td>79.18</td>
    <td>6.56034</td>
    <td>27.3099</td>
    <td>22.6 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B2</td>
    <td>81.74</td>
    <td>9.60494</td>
    <td>43.1219</td>
    <td>39.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B3</td>
    <td>82.98</td>
    <td>11.0042</td>
    <td>55.1367</td>
    <td>57.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B4</td>
    <td>83.57</td>
    <td>9.66407</td>
    <td>54.2462</td>
    <td>70.4 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B5</td>
    <td>84.75</td>
    <td>15.7091</td>
    <td>115.926</td>
    <td>140.8 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B6</td>
    <td>86.30</td>
    <td>21.226</td>
    <td>255.279</td>
    <td>268.4 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_5</td>
    <td>63.14</td>
    <td>3.67722</td>
    <td>6.66857</td>
    <td>6.7 M</td>
    <td rowspan="8">PP-LCNet is a lightweight backbone network developed by Baidu PaddlePaddle's vision team. It enhances model performance without increasing inference time, significantly surpassing other lightweight SOTA models.</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_25</td>
    <td>51.86</td>
    <td>2.65341</td>
    <td>5.81357</td>
    <td>5.5 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_35</td>
    <td>58.09</td>
    <td>2.7212</td>
    <td>6.28944</td>
    <td>5.9 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_75</td>
    <td>68.18</td>
    <td>3.91032</td>
    <td>8.06953</td>
    <td>8.4 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_0</td>
    <td>71.32</td>
    <td>3.84845</td>
    <td>9.23735</td>
    <td>10.5 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_5</td>
    <td>73.71</td>
    <td>3.97666</td>
    <td>12.3457</td>
    <td>16.0 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x2_0</td>
    <td>75.18</td>
    <td>4.07556</td>
    <td>16.2752</td>
    <td>23.2 M</td>
  </tr>
     <tr>
    <td>PP-LCNet_x2_5</td>
    <td>76.60</td>
    <td>4.06028</td>
    <td>21.5063</td>
    <td>32.1 M</td>
  </tr>
  <tr>

  <tr>
    <td>PP-LCNetV2_base</td>
    <td>77.05</td>
    <td>5.23428</td>
    <td>19.6005</td>
    <td>23.7 M</td>
    <td rowspan="3">The PP-LCNetV2 image classification model is the next-generation version of PP-LCNet, self-developed by Baidu PaddlePaddle's vision team. Based on PP-LCNet, it has undergone further optimization and improvements, primarily utilizing re-parameterization strategies to combine depthwise convolutions with varying kernel sizes and optimizing pointwise convolutions, Shortcuts, etc. Without using additional data, the PPLCNetV2_base model achieves over 77% Top-1 Accuracy on the ImageNet dataset for image classification, while maintaining an inference time of less than 4.4 ms on Intel CPU platforms.</td>
  </tr>
  <tr>
    <td>PP-LCNetV2_large </td>
    <td>78.51</td>
    <td>6.78335</td>
    <td>30.4378</td>
    <td>37.3 M</td>
  </tr>
  <tr>
    <td>PP-LCNetV2_small</td>
    <td>73.97</td>
    <td>3.89762</td>
    <td>13.0273</td>
    <td>14.6 M</td>
  </tr>
<tr>
<tr>
    <td>ResNet18_vd</td>
    <td>72.3</td>
    <td>3.53048</td>
    <td>31.3014</td>
    <td>41.5 M</td>
    <td rowspan="11">The ResNet series of models were introduced in 2015, winning the ILSVRC2015 competition with a top-5 error rate of 3.57%. This network innovatively proposed residual structures, which are stacked to construct the ResNet network. Experiments have shown that using residual blocks can effectively improve convergence speed and accuracy.</td>
  </tr>
  <tr>
    <td>ResNet18 </td>
    <td>71.0</td>
    <td>2.4868</td>
    <td>27.4601</td>
    <td>41.5 M</td>
  </tr>
  <tr>
    <td>ResNet34_vd</td>
    <td>76.0</td>
    <td>5.60675</td>
    <td>56.0653</td>
    <td>77.3 M</td>
  </tr>
    <tr>
    <td>ResNet34</td>
    <td>74.6</td>
    <td>4.16902</td>
    <td>51.925</td>
    <td>77.3 M</td>
  </tr>
  <tr>
    <td>ResNet50_vd</td>
    <td>79.1</td>
    <td>10.1885</td>
    <td>68.446</td>
    <td>90.8 M</td>
  </tr>
    <tr>
    <td>ResNet50</td>
    <td>76.5</td>
    <td>9.62383</td>
    <td>64.8135</td>
    <td>90.8 M</td>
  </tr>
     <tr>
    <td>ResNet101_vd</td>
    <td>80.2</td>
    <td>20.0563</td>
    <td>124.85</td>
    <td>158.4 M</td>
  </tr>
     <tr>
    <td>ResNet101</td>
    <td>77.6</td>
    <td>19.2297</td>
    <td>121.006</td>
    <td>158.4 M</td>
  </tr>
  <tr>
    <td>ResNet152_vd</td>
    <td>80.6</td>
    <td>29.6439</td>
    <td>181.678</td>
    <td>214.3 M</td>
  </tr>
    <tr>
    <td>ResNet152</td>
    <td>78.3</td>
    <td>30.0461</td>
    <td>177.707</td>
    <td>214.2 M</td>
  </tr>
     <tr>
    <td>ResNet200_vd</td>
    <td>80.9</td>
    <td>39.1628</td>
    <td>235.185</td>
    <td>266.0 M</td>
  </tr>
<tr>
  <tr>
    <td>StarNet-S1</td>
    <td>73.6</td>
    <td>9.895</td>
    <td>23.0465</td>
    <td>11.2 M</td>
    <td rowspan="4">StarNet focuses on exploring the untapped potential of "star operations" (i.e., element-wise multiplication) in network design. It reveals that star operations can map inputs to high-dimensional, nonlinear feature spaces, a process akin to kernel tricks but without the need to expand the network size. Consequently, StarNet, a simple yet powerful prototype network, is further proposed, demonstrating exceptional performance and low latency under compact network structures and limited computational resources.</td>
  </tr>
  <tr>
    <td>StarNet-S2 </td>
    <td>74.8</td>
    <td>7.91279</td>
    <td>21.9571</td>
    <td>14.3 M</td>
  </tr>
  <tr>
    <td>StarNet-S3</td>
    <td>77.0</td>
    <td>10.7531</td>
    <td>30.7656</td>
    <td>22.2 M</td>
  </tr>
    <tr>
    <td>StarNet-S4</td>
    <td>79.0</td>
    <td>15.2868</td>
    <td>43.2497</td>
    <td>28.9 M</td>
  </tr>
<tr>
  <tr>
    <td>SwinTransformer_base_patch4_window7_224</td>
    <td>83.37</td>
    <td>16.9848</td>
    <td>383.83</td>
    <td>310.5 M</td>
    <td rowspan="6">SwinTransformer is a novel vision Transformer network that can serve as a general-purpose backbone for computer vision tasks. SwinTransformer consists of a hierarchical Transformer structure represented by shifted windows. Shifted windows restrict self-attention computations to non-overlapping local windows while allowing cross-window connections, thereby enhancing network performance.</td>
  </tr>
  <tr>
    <td>SwinTransformer_base_patch4_window12_384</td>
    <td>84.17</td>
    <td>37.2855</td>
    <td>1178.63</td>
    <td>311.4 M</td>
  </tr>
  <tr>
    <td>SwinTransformer_large_patch4_window7_224</td>
    <td>86.19</td>
    <td>27.5498</td>
    <td>689.729</td>
    <td>694.8 M</td>
  </tr>
    <tr>
    <td>SwinTransformer_large_patch4_window12_384</td>
    <td>87.06</td>
    <td>74.1768</td>
    <td>2105.22</td>
    <td>696.1 M</td>
  </tr>
     <tr>
    <td>SwinTransformer_small_patch4_window7_224</td>
    <td>83.21</td>
    <td>16.3982</td>
    <td>285.56</td>
    <td>175.6 M</td>
  </tr>
       <tr>
    <td>SwinTransformer_tiny_patch4_window7_224</td>
    <td>81.10</td>
    <td>8.54846</td>
    <td>156.306</td>
    <td>100.1 M</td>
  </tr>


</table>

**Note: The above accuracy metrics refer to Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set. ****All model GPU inference times are based on NVIDIA Tesla T4 machines, with precision type FP32. CPU inference speeds are based on Intel® Xeon® Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.**
</details>

## 2. Quick Start
PaddleX provides pre-trained model pipelines that can be quickly experienced. You can experience the effects of the General Image Classification Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/100061/webUI) the effects of the General Image Classification Pipeline using the demo images provided by the official. For example:

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/02.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the model within the pipeline**.

### 2.2 Local Experience
Before using the General Image Classification Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

#### 2.2.1 Command Line Experience
A single command is all you need to quickly experience the image classification pipeline, Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the image classification pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default image classification pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to expand</summary>

```bash
paddlex --get_pipeline_config image_classification
```
After execution, the image classification pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config image_classification --save_path ./my_path
```

After obtaining the pipeline configuration file, replace `--pipeline` with the configuration file's save path to make the configuration file take effect. For example, if the configuration file's save path is `./image_classification.yaml`, simply execute:

```bash
paddlex --pipeline ./image_classification.yaml --input general_image_classification_001.jpg --device gpu:0
```
Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If you still specify parameters, the specified parameters will take precedence.

</details>

After running, the result will be:

```
{'input_path': 'general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': [0.62736, 0.03752, 0.03256, 0.0323, 0.03194], 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```
![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png)


The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

#### 2.2.2 Integration via Python Script
A few lines of code can complete the quick inference of the pipeline. Taking the general image classification pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_classification")

output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the visualization image of the result
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
The results obtained are the same as those obtained through the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
|`pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX. | `str` | None |
|`device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
|`use_hpip` | Whether to enable high-performance inference, which is only available when the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the image classification pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| `str` | Supports passing the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg). |
| `str` | Supports passing a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing a dictionary type, where the key needs to correspond to the specific task, such as "img" for the image classification task, and the value of the dictionary supports the above data types, e.g., `{"img": "/root/data1"}`. |
| `list` | Supports passing a list, where the list elements need to be the above data types, such as `[numpy.ndarray, numpy.ndarray]`, `["/root/data/img1.jpg", "/root/data/img2.jpg"]`, `["/root/data1", "/root/data2"]`, `[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/image_classification.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/image_classification.yaml")
output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualization image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end speedups. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

Below are the API references and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- Both the response body and the request body for POST requests are JSON data (JSON objects).
- When the request is processed successfully, the response status code is `200`, and the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Fixed as `0`.|
    |`errorMsg`|`string`|Error message. Fixed as `"Success"`.|

    The response body may also have a `result` property of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Same as the response status code.|
    |`errorMsg`|`string`|Error message.|

Operations provided by the service are as follows:

- **`infer`**

    Classify images.

    `POST /image-classification`

    - The request body properties are as follows:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        |`image`|`string`|The URL of an image file accessible by the service or the Base64 encoded result of the image file content.|Yes|
        |`inferenceParams`|`object`|Inference parameters.|No|

        The properties of `inferenceParams` are as follows:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        |`topK`|`integer`|Only the top `topK` categories with the highest scores will be retained in the results.|No|

    - When the request is processed successfully, the `result` of the response body has the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        |`categories`|`array`|Image category information.|
        |`image`|`string`|The image classification result image. The image is in JPEG format and encoded using Base64.|

        Each element in `categories` is an `object` with the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        |`id`|`integer`|Category ID.|
        |`name`|`string`|Category name.|
        |`score`|`number`|Category score.|

        An example of `result` is as follows:

        ```json
        {
          "categories": [
            {
              "id": 5,
              "name": "Rabbit",
              "score": 0.93
            }
          ],
          "image": "xxxxxx"
        }
        ```

</details>

<details>
<summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/image-classification"
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nCategories:")
print(result["categories"])
```

</details>
<details>
<summary>C++</summary>

```cpp
#include <iostream>
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

    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    auto response = client.Post("/image-classification", headers, body, "application/json");
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output image saved at " << outPutImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outPutImagePath << std::endl;
        }

        auto categories = result["categories"];
        std::cout << "\nCategories:" << std::endl;
        for (const auto& category : categories) {
            std::cout << category << std::endl;
        }
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        return 1;
    }

    return 0;
}
```

</details>

<details>
<summary>Java</summary>

```java
import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/image-classification";
        String imagePath = "./demo.jpg";
        String outputImagePath = "./out.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

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
```

</details>

<details>
<summary>Go</summary>

```go
package main

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

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    client := &http.Client{}
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
    err = json.Unmarshal([]byte(string(body)), &respData)
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
    fmt.Printf("Image saved at %s.jpg\n", outputImagePath)
    fmt.Println("\nCategories:")
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
    }
}
```

</details>

<details>
<summary>C#</summary>

```csharp
using System;
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

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

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
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/image-classification'
const imagePath = './demo.jpg'
const outputImagePath = "./out.jpg";

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
.then((response) => {
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nCategories:");
    console.log(result["categories"]);
})
.catch((error) => {
  console.log(error);
});
```

</details>
<details>
<summary>PHP</summary>

```php
<?php

$API_URL = "http://localhost:8080/image-classification";
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nCategories:\n";
print_r($result["categories"]);
?>
```

</details>

</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general image classification pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **data from your specific domain or application scenario** to improve the recognition performance of the general image classification pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general image classification pipeline includes an image classification module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/cv_modules/image_classification_en.md#四二次开发) section in the [Image Classification Module Development Tutorial](../../../module_usage/tutorials/cv_modules/image_classification_en.md) and use your private dataset to fine-tune the image classification model.

### 4.2 Model Application
After you have completed fine-tuning training using your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```yaml
......
Pipeline:
  model: PP-LCNet_x1_0  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Then, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference in the image classification pipeline, the Python command is:

```bash
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device npu:0
```
If you want to use the General Image Classification Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide_en.md).
