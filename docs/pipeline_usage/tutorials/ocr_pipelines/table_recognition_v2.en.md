---
comments: true
---

# General Table Recognition Pipeline v2 User Guide

## 1. Introduction to General Table Recognition Pipeline v2
Table recognition is a technology that automatically identifies and extracts table content and structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By using computer vision and machine learning algorithms, table recognition can convert complex table information into editable formats, facilitating further processing and analysis of data.

The General Table Recognition Pipeline v2 is designed to solve table recognition tasks by identifying tables in images and outputting them in HTML format. Unlike the previous version, this pipeline introduces two new modules: table classification and table cell detection, which are integrated with the table structure recognition module to complete the table recognition task. Based on this pipeline, precise predictions of tables can be achieved, covering a wide range of applications in general, manufacturing, finance, transportation, and other fields. The pipeline also provides flexible service deployment options, supporting various hardware and programming languages for integration. Moreover, it offers secondary development capabilities, allowing you to train and optimize models on your own dataset, which can then be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png"/>
<b>The General Table Recognition Pipeline v2 includes essential modules for table structure recognition, table classification, table cell localization, text detection, and text recognition, as well as optional modules for layout area detection, document image orientation classification, and text image correction.</b>

<b>If you prioritize model accuracy, choose a high-accuracy model; if you prioritize inference speed, choose a faster model; if you care about model size, choose a smaller model.</b>
<details><summary> 👉 Model List Details</summary>
<p><b>Table Recognition Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
<td rowspan="2">The SLANeXt series are the latest table structure recognition models developed by the PaddlePaddle Vision Team. Compared to SLANet and SLANet_plus, SLANeXt focuses on table structure recognition and has dedicated weights trained for wired and wireless tables, significantly improving the recognition ability for both types, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training Model</a></td>
</tr>
</table>

<p><b>Table Classification Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
<td>94.2</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>6.6M</td>
</tr>
</table>

<p><b>Table Cell Detection Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The PaddlePaddle Vision Team used RT-DETR-L as the base model and pre-trained it on a self-built table cell detection dataset, achieving good performance for both wired and wireless table cell detection.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>

<p><b>Text Detection Module Models:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>82.69</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>The server-side text detection model of PP-OCRv4, with higher precision, suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>77.79</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>The mobile text detection model of PP-OCRv4, with higher efficiency, suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>

<p><b>Text Recognition Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the ability to recognize some traditional Chinese characters, Japanese, and special characters, and can support the recognition of more than 15,000 characters. In addition to improving the text recognition capability related to documents, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>
The lightweight recognition model of PP-OCRv4 has high inference efficiency and can be deployed on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>The server-side model of PP-OCRv4 offers high inference accuracy and can be deployed on various types of servers.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>The ultra-lightweight English recognition model, trained based on the PP-OCRv4 recognition model, supports the recognition of English letters and numbers.</td>
</tr>
</table>

> ❗ The above list features the <b>4 core models</b> that the text recognition module primarily supports. In total, this module supports <b>18 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>

* <b>Chinese Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the recognition capabilities for some traditional Chinese characters, Japanese, and special characters. The number of recognizable characters is over 15,000. In addition to the improvement in document-related text recognition, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>The lightweight recognition model of PP-OCRv4 has high inference efficiency and can be deployed on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>The server-side model of PP-OCRv4 offers high inference accuracy and can be deployed on various types of servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>72.96</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td>PP-OCRv3’s lightweight recognition model is designed for high inference efficiency and can be deployed on a variety of hardware devices, including edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server text recognition model developed by the OpenOCR team of Fudan University's Visual and Learning Laboratory (FVL). It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the A list is 6% higher than that of PP-OCRv4.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">    The RepSVTR text recognition model is a mobile text recognition model based on SVTRv2. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the B list is 2.5% higher than that of PP-OCRv4, with the same inference speed.</td>
</tr>
</table>

* <b>English Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model supports the recognition of English and numbers.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv3 recognition model supports the recognition of English and numbers.</td>
</tr>
</table>

* <b>Multilingual Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>The ultra-lightweight Korean recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Korean and numbers. </td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>The ultra-lightweight Japanese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Japanese and numbers.</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>The ultra-lightweight Traditional Chinese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Traditional Chinese and numbers.</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>The ultra-lightweight Telugu recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Telugu and numbers.</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>The ultra-lightweight Kannada recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Kannada and numbers.</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>The ultra-lightweight Tamil recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Tamil and numbers.</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>The ultra-lightweight Latin recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Latin script and numbers.</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>The ultra-lightweight Arabic script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Arabic script and numbers.</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>
The ultra-lightweight cyrillic alphabet recognition model trained based on the PP-OCRv3 recognition model supports the recognition of cyrillic letters and numbers.</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M  </td>
<td>The ultra-lightweight Devanagari script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Devanagari script and numbers.</td>
</tr>
</table>
</details>
<p><b>Layout Region Detection Module Models (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5)（%）</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>A high-precision layout region localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports, based on RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>A layout region localization model with balanced accuracy and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports, based on PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>A high-efficiency layout region localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports, based on PicoDet-S.</td>
</tr>
</tbody>
</table>

> ❗ The above list includes the <b>3 core models</b> that are the focus of the layout detection module. The module supports a total of <b>11 full models</b>, including multiple predefined models with different categories. The complete list of models is as follows:
<details><summary> 👉Model List Details</summary>
* <b>Table Layout Detection Models</b>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset based on PicoDet-1x, capable of locating table areas.</td>
</tr>
</tbody></table>

* <b>3-Class Layout Detection Model, Including Tables, Images, and Stamps</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset based on PicoDet-S lightweight model, suitable for Chinese and English papers, magazines, and research reports.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>A balanced efficiency and accuracy layout area localization model trained on a self-built dataset based on PicoDet-L, suitable for Chinese and English papers, magazines, and research reports.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-built dataset based on RT-DETR-H, suitable for Chinese and English papers, magazines, and research reports.</td>
</tr>
</tbody></table>

* <b>5-Class English Document Area Detection Model, Including Text, Titles, Tables, Images, and Lists</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>A high-efficiency English document layout area localization model trained on the PubLayNet dataset based on PicoDet-1x.</td>
</tr>
</tbody></table>

<b>17-class layout detection model, covering 17 common categories, including: paragraph title, image, text, number, abstract, content, chart title, formula, table, table title, reference, document title, footnote, header, algorithm, footer, seal</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset for Chinese and English papers, magazines, and research reports based on the lightweight PicoDet-S model</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A balanced efficiency and accuracy layout region detection model trained on a self-built dataset for Chinese and English papers, magazines, and research reports based on PicoDet-L</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout region detection model trained on a self-built dataset for Chinese and English papers, magazines, and research reports based on RT-DETR-H</td>
</tr>
</tbody>
</table>
</details>
<p><b>Text Image Correction Module Model (Optional):</b></p>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>MS-SSIM (%)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>A high-precision text image rectification model</td>
</tr>
</tbody>
</table>

<p><b>Document Image Orientation Classification Module Model (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees</td>
</tr>
</tbody>
</table>

**Test Environment Description:**

- **Performance Test Environment**
  - **Test Datasets**
    - Document Image Orientation Classification Model: A self-built dataset by PaddleX, covering multiple scenarios such as certificates and documents, with 1,000 images.
    - Layout Region Detection Model: A self-built layout region detection dataset by PaddleOCR, containing 500 images of common document types such as Chinese and English papers, magazines, contracts, books, exams, and research reports.
    - Table Layout Detection Model: A self-built layout table region detection dataset by PaddleOCR, with 7,835 images of Chinese and English paper document types containing tables.
    - 3-Class Layout Detection Model: A self-built layout region detection dataset by PaddleOCR, containing 1,154 images of common document types such as Chinese and English papers, magazines, and research reports.
    - 5-Class English Document Region Detection Model: The evaluation dataset of [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet), containing 11,245 images of English documents. (Note: The link may not be accessible due to network issues or link validity. Please check the link and try again if necessary.)
    - 17-Class Region Detection Model: A self-built layout region detection dataset by PaddleOCR, containing 892 images of common document types such as Chinese and English papers, magazines, and research reports.
    - Table Structure Recognition Model: A self-built high-difficulty Chinese table recognition dataset by PaddleX.
    - Table Cell Detection Model: A self-built evaluation dataset by PaddleX.
    - Table Classification Model: A self-built evaluation dataset by PaddleX.
    - Text Detection Model: A self-built Chinese dataset by PaddleOCR, covering multiple scenarios such as street views, web images, documents, and handwriting, with 500 images for detection.
    - Chinese Recognition Model: A self-built Chinese dataset by PaddleOCR, covering multiple scenarios such as street views, web images, documents, and handwriting, with 11,000 images for text recognition.
    - ch_SVTRv2_rec: [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task](https://aistudio.baidu.com/competition/detail/1131/0/introduction) (Note: The link may not be accessible due to network issues or link validity. Please check the link and try again if necessary.)
    - ch_RepSVTR_rec: [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task](https://aistudio.baidu.com/competition/detail/1131/0/introduction) (Note: The link may not be accessible due to network issues or link validity. Please check the link and try again if necessary.)
    - English Recognition Model: A self-built English dataset by PaddleX.
    - Multilingual Recognition Model: A self-built multilingual dataset by PaddleX.
  - **Hardware Configuration**:
    - GPU: NVIDIA Tesla T4
    - CPU: Intel Xeon Gold 6271C @ 2.60GHz
    - Other Environment: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2

- **Inference Mode Description**

| Mode        | GPU Configuration                        | CPU Configuration | Acceleration Technology Combination                   |
|-------------|----------------------------------------|-------------------|---------------------------------------------------|
| Normal Mode | FP32 Precision / No TRT Acceleration   | FP32 Precision / 8 Threads | PaddleInference                                 |
| High-Performance Mode | Optimal combination of pre-selected precision types and acceleration strategies | FP32 Precision / 8 Threads | Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.) |


</details>

## 2. Quick Start
All models provided by PaddleX can be quickly experienced. You can use the command line or Python locally to experience the General Table Recognition Production Line v2.

### 2.1 Online Experience
Online experience is not supported at the moment.

### 2.2 Local Experience
Before using the General Table Recognition Production Line v2 locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.3 Command Line Experience
You can quickly experience the table recognition production line with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) (Note: The link may not be accessible due to network issues or link validity. Please check the link and try again if necessary.) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline table_recognition_v2 \
        --input table_recognition.jpg \
        --save_path ./output \
        --device gpu:0
```

<details><summary>👉 <b>After running, the result obtained is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'table_recognition.jpg', 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'Table', 'score': 0.9922188520431519, 'coordinate': [3.0127392, 0.14648987, 547.5102, 127.72023]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 'text_rec_score_thresh': 0, 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9951260685920715, 0.9943379759788513, 0.9968608021736145, 0.9978817105293274, 0.9985721111297607, 0.9616036415100098, 0.9977153539657593, 0.987593948841095, 0.9906861186027527, 0.9959743618965149, 0.9970152378082275, 0.9977849721908569, 0.9984450936317444], 'rec_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'rec_boxes': array([[234,   6, 316,  25],
       [ 38,  39,  73,  57],
       [122,  32, 201,  58],
       [227,  34, 346,  57],
       [351,  34, 391,  58],
       [417,  35, 534,  58],
       [ 34,  70,  78,  90],
       [287,  70, 328,  90],
       [454,  69, 496,  90],
       [ 17, 101,  95, 124],
       [144, 101, 178, 122],
       [278, 101, 338, 124],
       [448, 101, 503, 121]], dtype=int16)}, 'table_res_list': [{'cell_box_list': [array([3.18822289e+00, 1.46489874e-01, 5.46996138e+02, 3.08782365e+01]), array([  3.21032453,  31.1510637 , 110.20750237,  65.14108063]), array([110.18174553,  31.13076188, 213.00813103,  65.02860047]), array([212.96108818,  31.09959008, 404.19618034,  64.99535157]), array([404.08112907,  31.18304802, 547.00864983,  65.0847223 ]), array([  3.21772957,  65.0738733 , 110.33685875,  96.07921387]), array([110.23703575,  65.02486207, 213.08839226,  96.01378419]), array([213.06095695,  64.96230103, 404.28425407,  95.97141816]), array([404.23704338,  65.04879548, 547.01273918,  96.03654267]), array([  3.22793937,  96.08334137, 110.38572502, 127.08698823]), array([110.40586662,  96.10539795, 213.19943047, 127.07002045]), array([213.12627983,  96.0539148 , 404.42686272, 127.02842499]), array([404.33042717,  96.07251526, 547.01273918, 126.45088746])], 'pred_html': '<html><body><table><tr><td colspan="4">CRuncover</td></tr><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上 没想</td><td>江、整江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>', 'table_ocr_pred': {'rec_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9951260685920715, 0.9943379759788513, 0.9968608021736145, 0.9978817105293274, 0.9985721111297607, 0.9616036415100098, 0.9977153539657593, 0.987593948841095, 0.9906861186027527, 0.9959743618965149, 0.9970152378082275, 0.9977849721908569, 0.9984450936317444], 'rec_boxes': [array([234,   6, 316,  25], dtype=int16), array([38, 39, 73, 57], dtype=int16), array([122,  32, 201,  58], dtype=int16), array([227,  34, 346,  57], dtype=int16), array([351,  34, 391,  58], dtype=int16), array([417,  35, 534,  58], dtype=int16), array([34, 70, 78, 90], dtype=int16), array([287,  70, 328,  90], dtype=int16), array([454,  69, 496,  90], dtype=int16), array([ 17, 101,  95, 124], dtype=int16), array([144, 101, 178, 122], dtype=int16), array([278, 101, 338, 124], dtype=int16), array([448, 101, 503, 121], dtype=int16)]}}]}}
```

The result of the visualization is saved under `save_path`, and the visualization result of table recognition is as follows:
<img src="<url id="curl1qgonf4r89jtks8g" type="url" status="failed" title="" wc="0">https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/01.jpg</url> ">

</details>

### 2.2 Integration via Python Script
* The above command line is for a quick experience to view the results. Generally, in a project, integration through code is often required. You can complete the production line's fast inference with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="table_recognition.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps are executed:

(1) The `create_pipeline()` function is used to instantiate a Universal Table Recognition Pipeline v2 object. The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>The specific configuration information of the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline. It supports specifying specific GPU card numbers, such as "gpu:0", specific card numbers for other hardware, such as "npu:0", and CPU like "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, which is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the Universal Table Recognition Pipeline v2 object is called to perform inference prediction. This method returns a `generator`. Below are the parameters and descriptions of the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Optional</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>The data to be predicted, supports multiple input types, required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path of an image file or PDF file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">example</a> (Note: The link may not be accessible due to network issues or the validity of the URL. Please check the URL and try again if necessary.); <b>Local directory</b>, the directory must contain images to be predicted, such as the local path: <code>/root/data/</code> (Currently, prediction of PDF files in directories is not supported; PDF files must be specified with a specific file path).</li>
<li><b>List</b>: The elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
</tr>
</table>

</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Production line inference device</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line. During initialization, it will prioritize using the local GPU device 0. If not available, it will use the CPU device.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line, which is initialized as <code>True</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document unwarping module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line, which is initialized as <code>True</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limit for text detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line, which is initialized as <code>960</code>.</li>
</ul>
</td>
<td><code>None</code></td>
<td><code>text_det_limit_type</code></td>
<td>Type of image side length limit for text detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, while <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line, which is initialized as <code>max</code>.</li>
</ul>
</td>
<td><code>None</code></td>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold, in the output probability map, pixels with scores greater than this threshold will be considered as text pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line, which is <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold, the average score of all pixels within the detection box must be greater than this threshold for the result to be considered as a text area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the production line, which is <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion ratio, this method is used to expand the text area, the larger the value, the larger the expansion area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line, <code>2.0</code>, will be used.</li></li></ul></td>
<td><code>None</code></td>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold; text results with scores greater than this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line, <code>0.0</code>, will be used. That is, no threshold is set.</li></li></ul></td>
<td><code>None</code></td>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to use the layout detection module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Layout detection confidence threshold; only results with scores greater than this threshold will be output</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>dict</b>: The key is the int category ID, and the value is any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line, <code>0.5</code>, will be used.</li></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing after layout detection</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>The scaling factor for the side length of the detection box; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float|list|None</code></td>
<td>
<ul>
<li><b>float</b>: A floating-point number greater than 0, such as 1.1, indicating that the center of the detection box output by the model remains unchanged, and both the width and height are expanded by 1.1 times</li>
<li><b>list</b>: For example, [1.2, 1.5], indicating that the center of the detection box output by the model remains unchanged, the width is expanded by 1.2 times, and the height is expanded by 1.5 times</li>
<li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line will be used, initialized as 1.0</li>
</ul>
</td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The merging processing mode for the detection boxes output by the model; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>string|None</code></td>
<td>
<ul>
<li><b>large</b>: When set to large, for overlapping detection boxes in the model output, only the outermost largest box is retained, and overlapping inner boxes are removed.</li>
<li><b>small</b>: When set to small, for overlapping detection boxes in the model output, only the innermost smallest box is retained, and overlapping outer boxes are removed.</li>
<li><b>union</b>: No filtering of boxes is performed; both inner and outer boxes are retained</li>
<li><b>None</b>: If set to <code>None</code>, the default value of this parameter initialized on the production line will be used, initialized as <code>large</code></li>
</ul>
</td>
<td>None</td>
</tr>
</tr></table>
<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. If it is a directory, the saved file name will be consistent with the input file type</td>
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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the result as an xlsx file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the result as an html file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal, and the content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates which page of the PDF is currently being processed; otherwise, it is `None`.

    - `model_settings`: `(Dict[str, bool])` Configuration parameters for the production line models.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-line.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout detection sub-line.
        - `use_ocr_model`: `(bool)` Controls whether to enable the OCR sub-line.
    - `layout_det_res`: `(Dict[str, Union[List[numpy.ndarray], List[float]]])` Output results of the layout detection sub-module. Only exists when `use_layout_detection=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the layout detection module. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates which page of the PDF is currently being processed; otherwise, it is `None`.
        - `boxes`: `(List[Dict])` A list of detected layout seal region boxes, with each element in the list containing the following fields:
            - `cls_id`: `(int)` The class ID of the detected box.
            - `score`: `(float)` The confidence score of the detected box.
            - `coordinate`: `(List[float])` The coordinates of the four vertices of the detected box, in the order of x1, y1, x2, y2, representing the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.
    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` Output results of the document preprocessing sub-line. Only exists when `use_doc_preprocessor=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the preprocessing sub-line. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `model_settings`: `(Dict)` Model configuration parameters for the preprocessing sub-line.
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification.
            - `use_doc_unwarping`: `(bool)` Controls whether to enable document unwarping.
        - `angle`: `(int)` The predicted result of document orientation classification. When enabled, the values are [0,1,2,3], corresponding to [0°,90°,180°,270°]; when disabled, it is -1.

    - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with the array shape being (4, 2) and data type being int16.

    - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes.

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` The side length limit value for image preprocessing.
        - `limit_type`: `(str)` The processing method for side length limits.
        - `thresh`: `(float)` The confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` The expansion ratio for text detection boxes.
        - `text_type`: `(str)` The type of text detection, currently fixed as "general".

    - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results.

    - `rec_texts`: `(List[str])` A list of text recognition results, containing only texts with confidence scores above `text_rec_score_thresh`.

    - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, filtered by `text_rec_score_thresh`.

    - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes filtered by confidence score, in the same format as `dt_polys`.

    - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with shape (n, 4) and dtype int16. Each row represents the [x_min, y_min, x_max, y_max] coordinates of a rectangular box, where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` type will be converted to list format.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (It is not recommended to specify a specific file path directly, as multiple result images will be overwritten and only the last image will be retained.)
- Calling the `save_to_html()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}.html`. If a file is specified, it will be saved directly to that file. In the General Table Recognition Line v2, the HTML format of tables in the image will be written to the specified HTML file.
- Calling the `save_to_xlsx()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}.xlsx`. If a file is specified, it will be saved directly to that file. In the General Table Recognition Line v2, the Excel format of tables in the image will be written to the specified xlsx file.

* Additionally, it is also possible to obtain the visualization image with results and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format.</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualization image in <code>dict</code> format.</td>
</tr>
</table>

- The prediction result obtained through the `json` attribute is of dict type, and the content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary. The keys are `table_res_img`, `ocr_res_img`, `layout_res_img`, and `preprocessed_img`, corresponding to four `Image.Image` objects in order: the visualization image of table recognition results, the visualization image of OCR results, the visualization image of layout detection results, and the visualization image of image preprocessing. If a sub-module is not used, the corresponding result image will not be included in the dictionary.

Additionally, you can obtain the configuration file for the General Table Recognition Line v2 and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config table_recognition_v2 --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the General Table Recognition Production Line v2. Simply modify the `pipeline` parameter value in the `create_pipeline` method to the path of the production line configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/table_recognition_v2.yaml")

output = pipeline.predict(
    input="table_recognition.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the production line. If you want to change the initialization parameters for the General Table Recognition Production Line v2, you can directly modify the parameters in the configuration file and load the configuration file for prediction. At the same time, CLI prediction also supports passing in a configuration file, just specify the path of the configuration file with `--pipeline`.

## 3. Development Integration / Deployment
If the production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration / deployment.

If you need to apply the production line directly in your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed information on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common form of deployment in actual production environments. By encapsulating the inference functionality as a service, clients can access these services through network requests to obtain inference results. PaddleX supports various service-oriented deployment solutions for production lines. For detailed information on service-oriented deployment, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-oriented deployment and multi-language service invocation examples:

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
<th>Meaning</th>
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
<th>Meaning</th>
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
<p>Locate and recognize tables in the image.</p>
<p><code>POST /table-recognition</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of a server-accessible image or PDF file, or the Base64-encoded content of such files. For PDF files exceeding 10 pages, only the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>The type of file. <code>0</code> indicates a PDF file, <code>1</code> indicates an image file. If this attribute is not present in the request body, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_doc_orientation_classify</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_doc_unwarping</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_layout_detection</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrModel</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>use_ocr_model</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>layout_threshold</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Refer to the <code>layout_nms</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>Refer to the <code>layout_unclip_ratio</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refer to the <code>layout_merge_bboxes_mode</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Refer to the <code>text_det_limit_side_len</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refer to the <code>text_det_limit_type</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_thresh</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_box_thresh</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_det_unclip_ratio</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>text_rec_score_thresh</code> parameter description in the production line <code>predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tableRecResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the production line object's <code>predict</code> method, excluding the <code>input_path</code> field.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Refer to the <code>img</code> property description in the production line prediction results. The images are in JPEG format and are Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>

<details><summary>Multi-language Service Invocation Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/table-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["tableRecResults"]):
    print("Detected tables:")
    print(res["tables"])
    layout_img_path = f"layout_{i}.jpg"
    with open(layout_img_path, "wb") as f:
        f.write(base64.b64decode(res["layoutImage"]))
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output images saved at {layout_img_path} and {ocr_img_path}")
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing the devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method according to your needs to integrate the model into your AI application.

## 4. Secondary Development
If the default model weights provided by the General Table Recognition Pipeline v2 do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing model using <b>your own domain-specific or application data</b> to improve the recognition performance of the General Table Recognition Pipeline v2 in your scenario.

### 4.1 Model Fine-Tuning
Since the General Table Recognition Pipeline v2 consists of several modules, if the pipeline's performance is not satisfactory, the issue may arise from any one of these modules. You can analyze the images with poor recognition results to identify which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Module to Fine-Tune</th>
<th>Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Table classification error</td>
<td>Table Classification Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_classification.en.md">Link</a></td>
</tr>
<tr>
<td>Table cell positioning error</td>
<td>Table Cell Detection Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_cells_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Table structure recognition error</td>
<td>Table Structure Recognition Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md">Link</a></td>
</tr>
<tr>
<td>Failed to detect table area</td>
<td>Layout Area Detection Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/layout_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Text detection omission</td>
<td>Text Detection Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/text_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/text_recognition.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate image rotation correction</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text Image Correction Module</td>
<td>Not supported for fine-tuning</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding position in the pipeline configuration file.

```yaml
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PicoDet_layout_1x_table
    model_dir: null 

  TableClassification:
    module_name: table_classification
    model_name: PP-LCNet_x1_0_table_cls
    model_dir: null 

  WiredTableStructureRecognition:
    module_name: table_structure_recognition
    model_name: SLANeXt_wired
    model_dir: null 

  WirelessTableStructureRecognition:
    module_name: table_structure_recognition
    model_name: SLANeXt_wireless
    model_dir: null 

  WiredTableCellsDetection:
    module_name: table_cells_detection
    model_name: RT-DETR-L_wired_table_cell_det
    model_dir: null 

  WirelessTableCellsDetection:
    module_name: table_cells_detection
    model_name: RT-DETR-L_wireless_table_cell_det
    model_dir: null 

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null 

      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null

  GeneralOCR:
    pipeline_name: OCR
    text_type: general
    use_doc_preprocessor: False
    use_textline_orientation: False
    SubModules:
      TextDetection:
        module_name: text_detection
        model_name: PP-OCRv4_server_det
        model_dir: null 
        limit_side_len: 960
        limit_type: max
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 2.0

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv4_server_rec
        model_dir: null 
        batch_size: 1
        score_thresh: 0
```

Subsequently, refer to the command-line method or Python script method in [2.2 Local Experience](#22-Local-Experience) to load the modified production line configuration file.

## 5. Support for Multiple Hardware Devices
PaddleX supports a variety of mainstream hardware devices including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for OCR production line inference, the CLI command is:

```bash
paddlex --pipeline table_recognition_v2 \
        --input table_recognition.jpg \
        --save_path ./output \
        --device npu:0
```

Of course, you can also specify the hardware device when calling `create_pipeline()` or `predict()` in the Python script.

If you want to use the Universal Table Recognition Pipeline v2 on a wider range of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).