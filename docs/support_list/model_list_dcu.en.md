---
comments: true
---

# PaddleX Model List (Hygon DCU)

PaddleX incorporates multiple pipelines, each containing several modules, and each module encompasses various models. The specific models to use can be selected based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model storage size, select models with smaller storage sizes.

## Image Classification Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Top-1 Accuracy (%)</th>
<th>Model Storage Size (M)</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>ResNet18</td>
<td>71.0</td>
<td>41.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>ResNet34</td>
<td>74.6</td>
<td>77.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>ResNet50</td>
<td>76.5</td>
<td>90.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>ResNet101</td>
<td>77.6</td>
<td>158.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>ResNet152</td>
<td>78.3</td>
<td>214.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td>71.32</td>
<td>10.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set.</b>

## [Image Multi-label Classification Module](../module_usage/tutorials/cv_modules/image_multilabel_classification.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>Model Storage Size</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td>
<td>89.15</td>
<td>325.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-HGNetV2-B0_ML</td>
<td>80.98</td>
<td>39.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">Trained Model</a></td>
<tr>
<td>PP-HGNetV2-B4_ML</td>
<td>87.96</td>
<td>88.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-HGNetV2-B6_ML</td>
<td>91.06</td>
<td>286.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for the multi-label classification task mAP of [COCO2017](https://cocodataset.org/#home).</b>

## [Image Feature Module](../module_usage/tutorials/cv_modules/image_feature.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>recall@1（%）</th>
<th>Model Size</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>306.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for AliProducts recall@1。</b>

## Object Detection Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>Model Size (M)</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PicoDet-L</td>
<td>42.6</td>
<td>20.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PicoDet-M</td>
<td>37.5</td>
<td>16.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PicoDet-S</td>
<td>29.1</td>
<td>4.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PicoDet-XS</td>
<td>26.2</td>
<td>5.7M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td>52.9</td>
<td>185.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus-M</td>
<td>49.8</td>
<td>83.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td>43.7</td>
<td>28.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus-X</td>
<td>54.7</td>
<td>349.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>RT-DETR-R18</td>
<td>46.5</td>
<td>70.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>FCOS-ResNet50</td>
<td>39.6</td>
<td>124.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>YOLOX-N</td>
<td>26.1</td>
<td>3.4M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td>
<td>37.8</td>
<td>137.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>YOLOv3-DarkNet53</td>
<td>39.1</td>
<td>219.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar">Inference ModelInference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td>
<td>41.1</td>
<td>245.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are mAP(0.5:0.95) on the [COCO2017](https://cocodataset.org/#home) validation set.</b>

## [Small Object Detection Module](../module_usage/tutorials/cv_modules/small_object_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP（%）</th>
<th>Model Size</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE_plus_SOD-S</td>
<td>25.1</td>
<td>77.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td>
<td>31.9</td>
<td>325.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td>
<td>42.7</td>
<td>340.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for </b>[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset)<b> validation set mAP(0.5:0.95)。</b>

## [Semantic Segmentation Module](../module_usage/tutorials/cv_modules/semantic_segmentation.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mIoU (%)</th>
<th>Model Storage Size (M)</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td>
<td>80.36</td>
<td>94.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>Deeplabv3_Plus-R101</td>
<td>81.10</td>
<td>162.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-LiteSeg-T</td>
<td>73.10</td>
<td>28.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are mIoU on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.</b>

## [Abnormality Detection Module](../module_usage/tutorials/cv_modules/anomaly_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Avg（%）</th>
<th>Model Size</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>STFPM</td>
<td>96.2</td>
<td>21.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/STFPM_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on the </b>[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)<b> dataset using the average anomaly score.</b>

## [Face Detection Module](../module_usage/tutorials/cv_modules/face_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th style="text-align: center;">AP (%)<br/>Easy/Medium/Hard</th>
<th>Model Size</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PicoDet_LCNet_x2_5_face</td>
<td style="text-align: center;">93.7/90.7/68.1</td>
<td>28.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on the WIDER-FACE validation set with an input size of 640*640.</b>

## Text Detection Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Detection Hmean (%)</th>
<th>Model Size (M)</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.79</td>
<td>4.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.69</td>
<td>100.1M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, handwriting, and more scenarios, with 500 images for detection.</b>

## Text Recognition Module
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Recognition Avg Accuracy (%)</th>
<th>Model Size (M)</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.20</td>
<td>10.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td>79.20</td>
<td>71.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, handwriting, and more scenarios, with 11,000 images for text recognition.</b>

## [Time Series Forecasting Module](../module_usage/tutorials/time_series_modules/time_series_forecasting.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mse</th>
<th>mae</th>
<th>Model Size (M)</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>DLinear</td>
<td>0.382</td>
<td>0.394</td>
<td>72K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>NLinear</td>
<td>0.386</td>
<td>0.392</td>
<td>40K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/NLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/NLinear_pretrained.pdparams">Trained Model</a></td></tr>
<tr>
<td>RLinear</td>
<td>0.384</td>
<td>0.392</td>
<td>40K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RLinear_pretrained.pdparams">Trained Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar) dataset (evaluation results on the test set test.csv).</b>
