# Tutorial on Developing Image Classification Modules

## I. Overview
The image classification module is a crucial component in computer vision systems, responsible for categorizing input images. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. Typically, the image classification module receives an image as input and, through deep learning or other machine learning algorithms, classifies it into predefined categories based on its characteristics and content. For instance, in an animal recognition system, the image classification module might need to classify an input image as "cat," "dog," "horse," etc. The classification results from the image classification module are then output for use by other modules or systems.

## II. List of Supported Models
<details>
   <summary> 👉Details of Model List</summary>

<table>
  <tr>
    <th>Model</th>
    <th>Top-1 Accuracy (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
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

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md).

After installing the wheel package, you can complete image classification module inference with just a few lines of code. You can switch between models in this module freely, and you can also integrate the model inference of the image classification module into your project.

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better image classification models. Before using PaddleX to develop image classification models, please ensure that you have installed the relevant model training plugins for image classification in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/dF1VvOPZmZXXzn?t=mention&mt=doc&dt=doc).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and **only data that passes data validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use your own private dataset for subsequent model training, please refer to the [PaddleX Image Classification Task Module Data Annotation Guide](../../../data_annotations/cv_modules/image_classification.md).

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
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details>
  <summary>👉 <b>Validation Results Details (Click to Expand)</b></summary>

```bash
{
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
  "dataset_path": "./dataset/cls_flowers_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
```
The above validation results, with check_pass being True, indicate that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 102;
* `attributes.train_samples`: The number of training set samples in this dataset is 1020;
* `attributes.val_samples`: The number of validation set samples in this dataset is 1020;
* `attributes.train_sample_paths`: A list of relative paths to the visual samples in the training set of this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the visual samples in the validation set of this dataset;

Additionally, the dataset validation analyzes the sample number distribution across all classes in the dataset and generates a distribution histogram (histogram.png):

![](/tmp/images/modules/image_classification/01.png)
</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Image classification does not currently support data conversion.

**(2) Dataset Splitting**

The parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. The following are example explanations for some of the parameters in the configuration file:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. When set to `True`, the dataset format will be converted. The default is `False`;
    * `train_percent`: If re-splitting the dataset, you need to set the percentage of the training set, which should be an integer between 0-100, ensuring that the sum with `val_percent` equals 100;

For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, you need to modify the configuration file as follows:

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
```
Then execute the command:
```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
After the data splitting is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

These parameters also support being set through appending command line arguments:

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training
A single command can complete the model training. Taking the training of the image classification model PP-LCNet_x1_0 as an example:
```
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```

the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path of the training dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file parameter instructions for the corresponding task module of the model [PaddleX Common Model Configuration File Parameters](../../instructions/config_parameters_common.md).


<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves the model weight files, with the default being `output`. If you need to specify a save path, you can set it through the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in [PaddleX Model List (CPU/GPU)](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/0PKFjfhs0UN4Qs?t=mention&mt=doc&dt=doc). After completing the model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* `train_result.json`: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;
* `train.log`: Training log file, recording changes in model metrics and loss during training;
* `config.yaml`: Training configuration file, recording the hyperparameter configuration for this training session;
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;
</details>

## **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. Using PaddleX for model evaluation, a single command can complete the model evaluation:
```bash
python main.py -c  paddlex/configs/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed successfully and the model's evaluation metrics, including val.top1, val.top5;

</details>

### **4.4 Model Inference and Model Integration**
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-LCNet_x1_0.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

1.**Pipeline Integration**

The image classification module can be integrated into the [General Image Classification Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) of PaddleX. Simply replace the model path to update the image classification module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your obtained model.

2.**Module Integration**

The weights you produce can be directly integrated into the image classification module. You can refer to the Python example code in [Quick Integration](#三快速集成_en) and simply replace the model with the path to your trained model.
