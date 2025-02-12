---
comments: true
typora-copy-images-to: images
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/License-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows%2C%20Mac-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>



## 🔍 Introduction

PaddleX 3.0 is a low-code development tool for AI models built on the PaddlePaddle framework. It integrates numerous<b>ready-to-use pre-trained models</b>, enabling<b>full-process development</b>from model training to inference, supporting<b>a variety of mainstream hardware</b> both domestic and international, and aiding AI developers in industrial practice.

<style>
        .centered-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .centered-table th, .centered-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .centered-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .centered-table img {
            max-width: 100px;
            height: auto;
        }
        .img-table {
            width: 100%;
            margin: 0 auto;
            border-collapse: collapse;
            text-align: center;
        }
        .img-table th, .centered-table td {
            padding: 10px;
        }
        .img-table img {
            height: 126px;
            width: 180px;
            object-fit: cover;
        }
</style>

<table class="img-table">
        <tr>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_classification.html"><strong>Image Classification</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html"><strong>Multi-label Image Classification</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/object_detection.html"><strong>Object Detection</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html"><strong>Instance Segmentation</strong></a></th>
        </tr>
        <tr>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39"></td>
            <td><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/multilabel_cls.png"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182"></td>
        </tr>
        <tr>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html"><strong>Semantic Segmentation</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html"><strong>Image Anomaly Detection</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html"><strong>OCR</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html"><strong>Table Recognition</strong></a></th>
        </tr>
        <tr>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c"></td>
            <td><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/image_anomaly_detection.png"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa"></td>
        </tr>
        <tr>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.html"><strong>PP-ChatOCRv3-doc</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html"><strong>Time Series Forecasting</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html"><strong>Time Series Anomaly Detection</strong></a></th>
            <th><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html"><strong>Time Series Classification</strong></a></th>
        </tr>
        <tr>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6"></td>
            <td><img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e"></td>
        </tr>
    </table>

## 🛠️ Installation

!!! warning
    Please ensure you have a basic <b>Python runtime environment</b> before installing PaddleX (Note: Currently supports Python 3.8 to Python 3.10, with more Python versions being adapted).

### Installing PaddlePaddle

=== "CPU"
    ```bash
    python -m pip install paddlepaddle==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    ```
=== "CUDA 11.8"
    ```bash
    python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ```
=== "CUDA 12.3"
    ```bash
    python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
    ```

> ❗ For more PaddlePaddle Wheel versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html).


### Installing PaddleX

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl
```

> ❗ For more installation methods, please refer to the [PaddleX Installation Guide](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)

## 💻 Command Line Usage

A single command can quickly experience the production line effect, with a unified command line format as follows:

```bash
paddlex --pipeline [production line name] --input [input image] --device [running device]
```

You only need to specify three parameters:

* `pipeline`: The name of the production line
* `input`: The local path or URL of the input file to be processed (e.g., an image)
* `device`: The GPU number used (for example, `gpu:0` indicates using the 0th GPU), or you can choose to use CPU (`cpu`)


!!! example "OCR-related Pipelines CLI"

    === "OCR"

        ```bash
        paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': False}, 'angle': 0},'dt_polys': [array([[ 3, 10],[82, 10],[82, 33],[ 3, 33]], dtype=int16), ...], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, ...], 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.99*', ...], 'rec_scores': [0.8980069160461426,  ...], 'rec_polys': [array([[ 3, 10],[82, 10],[82, 33],[ 3, 33]], dtype=int16), ...], 'rec_boxes': array([[  3,  10,  82,  33], ...], dtype=int16)}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/03.png"></p>

    === "Table Recognition"

        ```bash
        paddlex --pipeline table_recognition \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'table_recognition.jpg', 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True, 'use_ocr_model': True}, 'doc_preprocessor_res': {'input_path': '0.jpg', 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 0}, 'layout_det_res': {'input_path': None, 'boxes': [{'cls_id': 0, 'label': 'Table', 'score': 0.9196816086769104, 'coordinate': [0, 8.614925, 550.9877, 132]}]}, 'overall_ocr_res': {'input_path': '0.jpg', 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[232,   0],
                    [318,   1],
                    [318,  24],
                    [232,  21]], dtype=int16), array([[32, 38],
                    [67, 38],
                    [67, 55],
                    [32, 55]], dtype=int16), array([[119,  34],
                    [196,  34],
                    [196,  57],
                    [119,  57]], dtype=int16), array([[222,  29],
                    [396,  31],
                    [396,  60],
                    [222,  58]], dtype=int16), array([[420,  30],
                    [542,  32],
                    [542,  61],
                    [419,  59]], dtype=int16), array([[29, 71],
                    [72, 71],
                    [72, 92],
                    [29, 92]], dtype=int16), array([[287,  72],
                    [329,  72],
                    [329,  93],
                    [287,  93]], dtype=int16), array([[458,  68],
                    [501,  71],
                    [499,  94],
                    [456,  91]], dtype=int16), array([[  9, 101],
                    [ 89, 103],
                    [ 89, 130],
                    [  8, 128]], dtype=int16), array([[139, 105],
                    [172, 105],
                    [172, 126],
                    [139, 126]], dtype=int16), array([[274, 103],
                    [339, 101],
                    [340, 128],
                    [275, 130]], dtype=int16), array([[451, 103],
                    [508, 103],
                    [508, 126],
                    [451, 126]], dtype=int16)], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 'text_rec_score_thresh': 0, 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上，没想', '江、江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9943075180053711, 0.9951075315475464, 0.9907732009887695, 0.9975494146347046, 0.9974043369293213, 0.9983242750167847, 0.991967499256134, 0.9898287653923035, 0.9961177110671997, 0.9975040555000305, 0.9986456632614136, 0.9987970590591431], 'rec_polys': [array([[232,   0],
                    [318,   1],
                    [318,  24],
                    [232,  21]], dtype=int16), array([[32, 38],
                    [67, 38],
                    [67, 55],
                    [32, 55]], dtype=int16), array([[119,  34],
                    [196,  34],
                    [196,  57],
                    [119,  57]], dtype=int16), array([[222,  29],
                    [396,  31],
                    [396,  60],
                    [222,  58]], dtype=int16), array([[420,  30],
                    [542,  32],
                    [542,  61],
                    [419,  59]], dtype=int16), array([[29, 71],
                    [72, 71],
                    [72, 92],
                    [29, 92]], dtype=int16), array([[287,  72],
                    [329,  72],
                    [329,  93],
                    [287,  93]], dtype=int16), array([[458,  68],
                    [501,  71],
                    [499,  94],
                    [456,  91]], dtype=int16), array([[  9, 101],
                    [ 89, 103],
                    [ 89, 130],
                    [  8, 128]], dtype=int16), array([[139, 105],
                    [172, 105],
                    [172, 126],
                    [139, 126]], dtype=int16), array([[274, 103],
                    [339, 101],
                    [340, 128],
                    [275, 130]], dtype=int16), array([[451, 103],
                    [508, 103],
                    [508, 126],
                    [451, 126]], dtype=int16)], 'rec_boxes': array([[232,   0, 318,  24],
                    [ 32,  38,  67,  55],
                    [119,  34, 196,  57],
                    [222,  29, 396,  60],
                    [419,  30, 542,  61],
                    [ 29,  71,  72,  92],
                    [287,  72, 329,  93],
                    [456,  68, 501,  94],
                    [  8, 101,  89, 130],
                    [139, 105, 172, 126],
                    [274, 101, 340, 130],
                    [451, 103, 508, 126]], dtype=int16)}, 'table_res_list': [{'cell_box_list': array([[  8.        ,   9.61492538, 532.        ,  26.61492538],
                    [  3.        ,  27.61492538, 104.        ,  65.61492538],
                    [109.        ,  28.61492538, 215.        ,  66.61492538],
                    [219.        ,  28.61492538, 396.        ,  64.61492538],
                    [396.        ,  29.61492538, 546.        ,  66.61492538],
                    [  1.        ,  65.61492538, 110.        ,  93.61492538],
                    [111.        ,  65.61492538, 215.        ,  94.61492538],
                    [220.        ,  66.61492538, 397.        ,  94.61492538],
                    [398.        ,  67.61492538, 544.        ,  94.61492538],
                    [  2.        ,  98.61492538, 111.        , 131.61492538],
                    [113.        ,  98.61492538, 216.        , 131.61492538],
                    [219.        ,  98.61492538, 400.        , 131.61492538],
                    [403.        ,  99.61492538, 545.        , 130.61492538]]), 'pred_html': '<html><body><table><tr><td colspan="4">CRuncover</td></tr><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上，没想</td><td>江、江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>', 'table_ocr_pred': {'rec_polys': [array([[232,   0],
                    [318,   1],
                    [318,  24],
                    [232,  21]], dtype=int16), array([[32, 38],
                    [67, 38],
                    [67, 55],
                    [32, 55]], dtype=int16), array([[119,  34],
                    [196,  34],
                    [196,  57],
                    [119,  57]], dtype=int16), array([[222,  29],
                    [396,  31],
                    [396,  60],
                    [222,  58]], dtype=int16), array([[420,  30],
                    [542,  32],
                    [542,  61],
                    [419,  59]], dtype=int16), array([[29, 71],
                    [72, 71],
                    [72, 92],
                    [29, 92]], dtype=int16), array([[287,  72],
                    [329,  72],
                    [329,  93],
                    [287,  93]], dtype=int16), array([[458,  68],
                    [501,  71],
                    [499,  94],
                    [456,  91]], dtype=int16), array([[  9, 101],
                    [ 89, 103],
                    [ 89, 130],
                    [  8, 128]], dtype=int16), array([[139, 105],
                    [172, 105],
                    [172, 126],
                    [139, 126]], dtype=int16), array([[274, 103],
                    [339, 101],
                    [340, 128],
                    [275, 130]], dtype=int16), array([[451, 103],
                    [508, 103],
                    [508, 126],
                    [451, 126]], dtype=int16)], 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上，没想', '江、江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9943075180053711, 0.9951075315475464, 0.9907732009887695, 0.9975494146347046, 0.9974043369293213, 0.9983242750167847, 0.991967499256134, 0.9898287653923035, 0.9961177110671997, 0.9975040555000305, 0.9986456632614136, 0.9987970590591431], 'rec_boxes': [array([232,   0, 318,  24], dtype=int16), array([32, 38, 67, 55], dtype=int16), array([119,  34, 196,  57], dtype=int16), array([222,  29, 396,  60], dtype=int16), array([419,  30, 542,  61], dtype=int16), array([29, 71, 72, 92], dtype=int16), array([287,  72, 329,  93], dtype=int16), array([456,  68, 501,  94], dtype=int16), array([  8, 101,  89, 130], dtype=int16), array([139, 105, 172, 126], dtype=int16), array([274, 101, 340, 130], dtype=int16), array([451, 103, 508, 126], dtype=int16)]}}]}}
                ```

            === "img"
                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png"></p>

    === "Table Recognition v2"

        ```bash
        paddlex --pipeline table_recognition_v2 \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
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

            === "img"
                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/03.png"></p>


    === "Layout Parsing"

        ```bash
        paddlex --pipeline layout_parsing \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0 \
        ```

        ??? question "What's the result"
            ```bash
                {'res': {'input_path': 'demo_paper.png', 'model_settings': {'use_doc_preprocessor': True, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True}, 'parsing_res_list': [{'layout_bbox': [46.905365, 44.05746, 565.6911, 217.74211], 'image': array([[[255, ..., 255],
                        ...,
                        [188, ..., 175]],

                    ...,

                    [[255, ..., 255],
                        ...,
                        [255, ..., 255]]], dtype=uint8), 'image_text': 'Efficient Hybrid Encoder\nCaevIst s1\nCaaveJs2\nCCFF\n\nFusion\n00□0□000000', 'layout': 'single'}, {'layout_bbox': [42.178703, 227.34215, 570.1248, 284.377], 'text': 'Figure 4, Overview of RT-DETR. We feed the features from the last three stages of the backbone into the encoder. The eficient hybrid\nencoder transforms multi-scale features into a sequence of image features through the Attention-based Intra-scale Feature Interaction (AIFI)\nfeatures to serve as initial object queries for the decoder, Finaly, the decoder with auxiliary prediction heads iteratively optimizes object\nand the CNN-based Cross-scale Feature Fusion (CCFF), Then, the uncertainty-minimal query selection selects a fixed number of encoder\nqueries to generate categories and boxes.', 'layout': 'single'}, {'layout_bbox': [53.227856, 294.16644, 283.854, 396.24164], 'image': array([[[255, ..., 255],
                        ...,
                        [255, ..., 255]],

                    ...,

                    [[255, ..., 255],
                        ...,
                        [255, ..., 255]]], dtype=uint8), 'image_text': '1x1Conv\nFusion\nC\nNX\n1x1Conv\nRepBlock\nCConcatenate\nElement-wise add\nFlatten', 'layout': 'double'}, {'layout_bbox': [99.52045, 401.63477, 240.27536, 411.01334], 'text': 'Figure 5. The fusion block in CCFF', 'layout': 'double'}, {'layout_bbox': [41.776196, 418.61166, 296.82672, 554.4149], 'text': 'D, Ds, not only significantly reduces latency (35% faster),\nbut also improves accuracy (0.4% AP higher). CCFF is opti-\nmized based on the cross-scale fusion module, which inserts\nseveral fusion blocks consisting of convolutional layers into\nthe fusion path. The role of the fusion block is to fuse two\nadjacent scale features into a new feature, and its structure is\nillustrated in Figure 5. The fusion block contains two 1 × 1\nconvolutions to adjust the number of channels, N RepBlocks\ncomposed of RepConv [8] are used for feature fusion, and\nthe two-path outputs are fused by element-wise add. We\nformulate the calculation of the hybrid encoder as:', 'layout': 'double'}, {'layout_bbox': [92.092064, 563.1221, 297.57217, 607.2598], 'formula': '\\begin{array}{r l}{\\mathcal{Q}}&{=\\mathcal{K}=\\mathcal{V}=\\mathtt{F l a t t e r n}(\\mathcal{S}_{5}),}\\\\ {\\mathcal{F}_{5}}&{=\\mathtt{R e s h a p e}(\\mathtt{A I F I}(\\mathcal{Q},\\mathcal{K},\\mathcal{V})),\\qquad\\quad(1)}\\\\ {\\mathcal{O}}&{=\\mathtt{C C F F}(\\{\\mathcal{S}_{3},\\mathcal{S}_{4},\\mathcal{F}_{5}\\}),}\\end{array}', 'layout': 'double'}, {'layout_bbox': [41.275124, 613.64154, 298.0696, 636.9947], 'text': 'where Reshape represents restoring the shape of the flat-\ntened feature to the same shape as S5', 'layout': 'double'}, {'layout_bbox': [41.01993, 645.3085, 253.87302, 656.61505], 'text': '4.3.Uncertainty-minimal Ouery Selection', 'layout': 'double'}, {'layout_bbox': [39.780045, 664.1547, 296.5302, 724.974], 'text': 'To reduce the difficulty of optimizing object queries in\nDETR, several subsequent works [42, 44, 45] propose query\nselection schemes, which have in common that they use the\nconfidence score to select the top K features from the en-\ncoder to initialize object queries (or just position queries).', 'layout': 'double'}, {'layout_bbox': [316.3008, 289.54156, 573.4635, 415.4659], 'text': 'The confidence score represents the likelihood that the fea-\nture includes foreground objects. Nevertheless, the detector\nare required to simultaneously model the category and loca-\ntion of objects, both of which determine the quality of the\nfeatures. Hence, the performance score of the feature is a la-\ntent variable that is jointly correlated with both classification\nand localization. Based on the analysis, the current query\nselection lead to a considerable level of uncertainty in the\nselected features, resulting in sub-optimal initialization for\nthe decoder and hindering the performance of the detector.', 'layout': 'double'}, {'layout_bbox': [316.1587, 417.67807, 575.0031, 541.93054], 'text': 'To address this problem, we propose the uncertainty mini-\nmal query selection scheme, which explicitly constructs and\noptimizes the epistemic uncertainty to model the joint latent\nvariable of encoder features, thereby providing high-quality\nqueries for the decoder. Specifically, the feature uncertainty\nL/ is defined as the discrepancy between the predicted dis-\ntributions of localization P and classification C in Eq. (2).\nTo minimize the uncertainty of the queries, we integrate\nthe uncertainty into the loss function for the gradient-based\noptimization in Eq. (3).', 'layout': 'double'}, {'layout_bbox': [343.82712, 551.06995, 573.45465, 589.9438], 'formula': '\\begin{array}{r l r}{\\mathcal{U}(\\hat{\\mathcal{X}})=\\|\\mathcal{P}(\\hat{\\mathcal{X}})-\\mathcal{C}(\\hat{\\mathcal{X}})\\|,\\hat{\\mathcal{X}}\\in\\mathbb{R}^{D}}&{{}(2)}&{}\\\\ {\\mathcal{L}(\\hat{\\mathcal{X}},\\hat{\\mathcal{Y}},\\mathcal{Y})=\\mathcal{L}_{t o x}(\\hat{\\mathbf{b}},\\mathbf{b})+\\mathcal{L}_{c l s}(\\mathcal{U}(\\hat{\\mathcal{X}}),\\hat{\\mathbf{c}},\\mathbf{c})}&{{}(3)}\\end{array}', 'layout': 'double'}, {'layout_bbox': [316.74704, 598.45776, 573.39526, 636.35236], 'text': 'where  and y denote the prediction and ground truth.\n= (e, b), C and b represent the category and bounding\nbox respectively, X represent the encoder feature.', 'layout': 'double'}, {'layout_bbox': [315.35437, 638.09393, 572.0008, 724.53687], 'text': 'Effectiveness analysis. To analyze the effectiveness of thc\nuncertainty-minimal query selection, we visualize the clas-\nsification scores and IoU scores of the selected features on\nCOCO va1.2017, Figure 6. We draw the scatterplot with\nclassification scores greater than 0.5. The purple and green\nwith uncertainty-minimal query selection and vanilla query\ndots represent the selected features from the model trained', 'layout': 'double'}], 'doc_preprocessor_res': {'input_path': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 0}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 1, 'label': 'image', 'score': 0.9874590635299683, 'coordinate': [46.905365, 44.05746, 565.6911, 217.74211]}, {'cls_id': 2, 'label': 'text', 'score': 0.9869957566261292, 'coordinate': [41.776196, 418.61166, 296.82672, 554.4149]}, {'cls_id': 2, 'label': 'text', 'score': 0.9792540073394775, 'coordinate': [39.780045, 664.1547, 296.5302, 724.974]}, {'cls_id': 2, 'label': 'text', 'score': 0.9792136549949646, 'coordinate': [316.3008, 289.54156, 573.4635, 415.4659]}, {'cls_id': 2, 'label': 'text', 'score': 0.9789648652076721, 'coordinate': [316.1587, 417.67807, 575.0031, 541.93054]}, {'cls_id': 1, 'label': 'image', 'score': 0.9786934852600098, 'coordinate': [53.227856, 294.16644, 283.854, 396.24164]}, {'cls_id': 2, 'label': 'text', 'score': 0.9765349626541138, 'coordinate': [315.35437, 638.09393, 572.0008, 724.53687]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9575827717781067, 'coordinate': [42.178703, 227.34215, 570.1248, 284.377]}, {'cls_id': 2, 'label': 'text', 'score': 0.9554654359817505, 'coordinate': [41.275124, 613.64154, 298.0696, 636.9947]}, {'cls_id': 7, 'label': 'formula', 'score': 0.951255738735199, 'coordinate': [92.092064, 563.1221, 297.57217, 607.2598]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9501133561134338, 'coordinate': [343.82712, 551.06995, 573.45465, 589.9438]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9381633400917053, 'coordinate': [99.52045, 401.63477, 240.27536, 411.01334]}, {'cls_id': 2, 'label': 'text', 'score': 0.9283379316329956, 'coordinate': [316.74704, 598.45776, 573.39526, 636.35236]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9257320761680603, 'coordinate': [41.01993, 645.3085, 253.87302, 656.61505]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[169,  50],
                        ...,
                        [169,  62]],

                    ...,

                    [[ 39, 711],
                        ...,
                        [ 39, 726]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['Efficient Hybrid Encoder', 'CaevIst s1', 'CaaveJs2', 'CCFF', '', 'Fusion', '00□0□000000', 'Figure 4, Overview of RT-DETR. We feed the features from the last three stages of the backbone into the encoder. The eficient hybrid', 'encoder transforms multi-scale features into a sequence of image features through the Attention-based Intra-scale Feature Interaction (AIFI)', 'features to serve as initial object queries for the decoder, Finaly, the decoder with auxiliary prediction heads iteratively optimizes object', 'and the CNN-based Cross-scale Feature Fusion (CCFF), Then, the uncertainty-minimal query selection selects a fixed number of encoder', 'queries to generate categories and boxes.', '1x1Conv', 'Fusion', 'The confidence score represents the likelihood that the fea-', 'C', 'ture includes foreground objects. Nevertheless, the detector', 'are required to simultaneously model the category and loca-', 'NX', 'tion of objects, both of which determine the quality of the', '1x1Conv', 'features. Hence, the performance score of the feature is a la-', 'RepBlock', 'tent variable that is jointly correlated with both classification', 'and localization. Based on the analysis, the current query', 'CConcatenate', 'Element-wise add', 'Flatten', 'selection lead to a considerable level of uncertainty in the', 'selected features, resulting in sub-optimal initialization for', 'Figure 5. The fusion block in CCFF', 'the decoder and hindering the performance of the detector.', 'D, Ds, not only significantly reduces latency (35% faster),', 'To address this problem, we propose the uncertainty mini-', 'but also improves accuracy (0.4% AP higher). CCFF is opti-', 'mal query selection scheme, which explicitly constructs and', 'mized based on the cross-scale fusion module, which inserts', 'optimizes the epistemic uncertainty to model the joint latent', 'several fusion blocks consisting of convolutional layers into', 'variable of encoder features, thereby providing high-quality', 'the fusion path. The role of the fusion block is to fuse two', 'queries for the decoder. Specifically, the feature uncertainty', 'adjacent scale features into a new feature, and its structure is', 'L/ is defined as the discrepancy between the predicted dis-', 'illustrated in Figure 5. The fusion block contains two 1 × 1', 'tributions of localization P and classification C in Eq. (2).', 'convolutions to adjust the number of channels, N RepBlocks', 'To minimize the uncertainty of the queries, we integrate', 'composed of RepConv [8] are used for feature fusion, and', 'the uncertainty into the loss function for the gradient-based', 'the two-path outputs are fused by element-wise add. We', 'optimization in Eq. (3).', 'formulate the calculation of the hybrid encoder as:', 'u(x)=P(x）-C(x）,x∈RD', '(2)', '=K =V =F1atten(Ss),', 'F = Reshape(AIFI(Q,K,V)),', '(1)', 'C(x.y)= Lo（b,b）+Cc(u(x）,e,c)（3)', 'O=CCFF（{S,S,F}）', 'where  and y denote the prediction and ground truth.', 'where Reshape represents restoring the shape of the flat-', '= (e, b), C and b represent the category and bounding', 'tened feature to the same shape as S5', 'box respectively, X represent the encoder feature.', '4.3.Uncertainty-minimal Ouery Selection', 'Effectiveness analysis. To analyze the effectiveness of thc', 'uncertainty-minimal query selection, we visualize the clas-', 'To reduce the difficulty of optimizing object queries in', 'sification scores and IoU scores of the selected features on', 'DETR, several subsequent works [42, 44, 45] propose query', 'COCO va1.2017, Figure 6. We draw the scatterplot with', 'selection schemes, which have in common that they use the', 'classification scores greater than 0.5. The purple and green', 'confidence score to select the top K features from the en-', 'with uncertainty-minimal query selection and vanilla query', 'dots represent the selected features from the model trained', 'coder to initialize object queries (or just position queries).'], 'rec_scores': array([0.95921248, ..., 0.99757016]), 'rec_polys': array([[[169,  50],
                        ...,
                        [169,  62]],

                    ...,

                    [[ 39, 711],
                        ...,
                        [ 39, 726]]], dtype=int16), 'rec_boxes': array([[169, ...,  62],
                    ...,
                    [ 39, ..., 726]], dtype=int16)}, 'text_paragraphs_ocr_res': {'rec_polys': array([[[169,  50],
                        ...,
                        [169,  62]],

                    ...,

                    [[ 39, 711],
                        ...,
                        [ 39, 726]]], dtype=int16), 'rec_texts': ['Efficient Hybrid Encoder', 'CaevIst s1', 'CaaveJs2', 'CCFF', '', 'Fusion', '00□0□000000', 'Figure 4, Overview of RT-DETR. We feed the features from the last three stages of the backbone into the encoder. The eficient hybrid', 'encoder transforms multi-scale features into a sequence of image features through the Attention-based Intra-scale Feature Interaction (AIFI)', 'features to serve as initial object queries for the decoder, Finaly, the decoder with auxiliary prediction heads iteratively optimizes object', 'and the CNN-based Cross-scale Feature Fusion (CCFF), Then, the uncertainty-minimal query selection selects a fixed number of encoder', 'queries to generate categories and boxes.', '1x1Conv', 'Fusion', 'The confidence score represents the likelihood that the fea-', 'C', 'ture includes foreground objects. Nevertheless, the detector', 'are required to simultaneously model the category and loca-', 'NX', 'tion of objects, both of which determine the quality of the', '1x1Conv', 'features. Hence, the performance score of the feature is a la-', 'RepBlock', 'tent variable that is jointly correlated with both classification', 'and localization. Based on the analysis, the current query', 'CConcatenate', 'Element-wise add', 'Flatten', 'selection lead to a considerable level of uncertainty in the', 'selected features, resulting in sub-optimal initialization for', 'Figure 5. The fusion block in CCFF', 'the decoder and hindering the performance of the detector.', 'D, Ds, not only significantly reduces latency (35% faster),', 'To address this problem, we propose the uncertainty mini-', 'but also improves accuracy (0.4% AP higher). CCFF is opti-', 'mal query selection scheme, which explicitly constructs and', 'mized based on the cross-scale fusion module, which inserts', 'optimizes the epistemic uncertainty to model the joint latent', 'several fusion blocks consisting of convolutional layers into', 'variable of encoder features, thereby providing high-quality', 'the fusion path. The role of the fusion block is to fuse two', 'queries for the decoder. Specifically, the feature uncertainty', 'adjacent scale features into a new feature, and its structure is', 'L/ is defined as the discrepancy between the predicted dis-', 'illustrated in Figure 5. The fusion block contains two 1 × 1', 'tributions of localization P and classification C in Eq. (2).', 'convolutions to adjust the number of channels, N RepBlocks', 'To minimize the uncertainty of the queries, we integrate', 'composed of RepConv [8] are used for feature fusion, and', 'the uncertainty into the loss function for the gradient-based', 'the two-path outputs are fused by element-wise add. We', 'optimization in Eq. (3).', 'formulate the calculation of the hybrid encoder as:', 'where  and y denote the prediction and ground truth.', 'where Reshape represents restoring the shape of the flat-', '= (e, b), C and b represent the category and bounding', 'tened feature to the same shape as S5', 'box respectively, X represent the encoder feature.', '4.3.Uncertainty-minimal Ouery Selection', 'Effectiveness analysis. To analyze the effectiveness of thc', 'uncertainty-minimal query selection, we visualize the clas-', 'To reduce the difficulty of optimizing object queries in', 'sification scores and IoU scores of the selected features on', 'DETR, several subsequent works [42, 44, 45] propose query', 'COCO va1.2017, Figure 6. We draw the scatterplot with', 'selection schemes, which have in common that they use the', 'classification scores greater than 0.5. The purple and green', 'confidence score to select the top K features from the en-', 'with uncertainty-minimal query selection and vanilla query', 'dots represent the selected features from the model trained', 'coder to initialize object queries (or just position queries).'], 'rec_scores': array([0.95921248, ..., 0.99757016]), 'rec_boxes': array([[169, ...,  62],
                    ...,
                    [ 39, ..., 726]], dtype=int16)}, 'formula_res_list': [{'input_path': None, 'page_index': None, 'rec_formula': '\\begin{array}{r l}{\\mathcal{Q}}&{=\\mathcal{K}=\\mathcal{V}=\\mathtt{F l a t t e r n}(\\mathcal{S}_{5}),}\\\\ {\\mathcal{F}_{5}}&{=\\mathtt{R e s h a p e}(\\mathtt{A I F I}(\\mathcal{Q},\\mathcal{K},\\mathcal{V})),\\qquad\\quad(1)}\\\\ {\\mathcal{O}}&{=\\mathtt{C C F F}(\\{\\mathcal{S}_{3},\\mathcal{S}_{4},\\mathcal{F}_{5}\\}),}\\end{array}', 'formula_region_id': 1, 'dt_polys': [92.092064, 563.1221, 297.57217, 607.2598]}, {'input_path': None, 'page_index': None, 'rec_formula': '\\begin{array}{r l r}{\\mathcal{U}(\\hat{\\mathcal{X}})=\\|\\mathcal{P}(\\hat{\\mathcal{X}})-\\mathcal{C}(\\hat{\\mathcal{X}})\\|,\\hat{\\mathcal{X}}\\in\\mathbb{R}^{D}}&{{}(2)}&{}\\\\ {\\mathcal{L}(\\hat{\\mathcal{X}},\\hat{\\mathcal{Y}},\\mathcal{Y})=\\mathcal{L}_{t o x}(\\hat{\\mathbf{b}},\\mathbf{b})+\\mathcal{L}_{c l s}(\\mathcal{U}(\\hat{\\mathcal{X}}),\\hat{\\mathbf{c}},\\mathbf{c})}&{{}(3)}\\end{array}', 'formula_region_id': 2, 'dt_polys': [343.82712, 551.06995, 573.45465, 589.9438]}]}}
                ```

    === "Formula Recognition"

        ```bash
        paddlex --pipeline formula_recognition \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png \
        --use_layout_detection True \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --layout_threshold 0.5 \
        --layout_nms True \
        --layout_unclip_ratio  1.0 \
        --layout_merge_bboxes_mode large \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'general_formula_recognition.png', 'model_settings': {'use_doc_preprocessor': False,'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9778407216072083, 'coordinate': [271.257, 648.50824, 1040.2291, 774.8482]}, ...]}, 'formula_res_list': [{'rec_formula': '\\small\\begin{aligned}{p(\\mathbf{x})=c(\\mathbf{u})\\prod_{i}p(x_{i}).}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([553.0718, 802.0996, 758.75635, 853.093],)}, ...]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04.png"></p>

    === "Seal Text Recognition"

        ```bash
        paddlex --pipeline seal_recognition \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --device gpu:0 \
        --save_path ./output
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                        {'res': {'input_path': 'seal_text_det.png', 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 16, 'label': 'seal', 'score': 0.975529670715332, 'coordinate': [6.191284, 0.16680908, 634.39325, 628.85345]}]}, 'seal_res_list': [{'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[320,  38],
                            [479,  92],
                            [483,  94],
                            [486,  97],
                            [579, 226],
                            [582, 230],
                            [582, 235],
                            [584, 383],
                            [584, 388],
                            [582, 392],
                            [578, 396],
                            [573, 398],
                            [566, 398],
                            [502, 380],
                            [497, 377],
                            [494, 374],
                            [491, 369],
                            [491, 366],
                            [488, 259],
                            [424, 172],
                            [318, 136],
                            [251, 154],
                            [200, 174],
                            [137, 260],
                            [133, 366],
                            [132, 370],
                            [130, 375],
                            [126, 378],
                            [123, 380],
                            [ 60, 398],
                            [ 55, 398],
                            [ 49, 397],
                            [ 45, 394],
                            [ 43, 390],
                            [ 41, 383],
                            [ 43, 236],
                            [ 44, 230],
                            [ 45, 227],
                            [141,  96],
                            [144,  93],
                            [148,  90],
                            [311,  38],
                            [315,  38]]), array([[461, 347],
                            [465, 350],
                            [468, 354],
                            [470, 360],
                            [470, 425],
                            [469, 429],
                            [467, 433],
                            [462, 437],
                            [456, 439],
                            [169, 439],
                            [165, 439],
                            [160, 436],
                            [157, 432],
                            [155, 426],
                            [154, 360],
                            [155, 356],
                            [158, 352],
                            [161, 348],
                            [168, 346],
                            [456, 346]]), array([[439, 445],
                            [441, 447],
                            [443, 451],
                            [444, 453],
                            [444, 497],
                            [443, 502],
                            [440, 504],
                            [437, 506],
                            [434, 507],
                            [189, 505],
                            [184, 504],
                            [182, 502],
                            [180, 498],
                            [179, 496],
                            [181, 453],
                            [182, 449],
                            [184, 446],
                            [188, 444],
                            [434, 444]]), array([[158, 468],
                            [199, 502],
                            [242, 522],
                            [299, 534],
                            [339, 532],
                            [373, 526],
                            [417, 508],
                            [459, 475],
                            [462, 474],
                            [467, 474],
                            [472, 476],
                            [502, 507],
                            [503, 510],
                            [504, 515],
                            [503, 518],
                            [501, 521],
                            [452, 559],
                            [450, 560],
                            [391, 584],
                            [390, 584],
                            [372, 590],
                            [370, 590],
                            [305, 596],
                            [302, 596],
                            [224, 581],
                            [221, 580],
                            [164, 553],
                            [162, 551],
                            [114, 509],
                            [112, 507],
                            [111, 503],
                            [112, 498],
                            [114, 496],
                            [146, 468],
                            [149, 466],
                            [154, 466]])], 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.2, 'box_thresh': 0.6, 'unclip_ratio': 0.5}, 'text_type': 'seal', 'textline_orientation_angles': [-1, -1, -1, -1], 'text_rec_score_thresh': 0, 'rec_texts': ['天津君和缘商贸有限公司', '发票专用章', '吗繁物', '5263647368706'], 'rec_scores': [0.9934046268463135, 0.9999403953552246, 0.998250424861908, 0.9913849234580994], 'rec_polys': [array([[320,  38],
                            [479,  92],
                            [483,  94],
                            [486,  97],
                            [579, 226],
                            [582, 230],
                            [582, 235],
                            [584, 383],
                            [584, 388],
                            [582, 392],
                            [578, 396],
                            [573, 398],
                            [566, 398],
                            [502, 380],
                            [497, 377],
                            [494, 374],
                            [491, 369],
                            [491, 366],
                            [488, 259],
                            [424, 172],
                            [318, 136],
                            [251, 154],
                            [200, 174],
                            [137, 260],
                            [133, 366],
                            [132, 370],
                            [130, 375],
                            [126, 378],
                            [123, 380],
                            [ 60, 398],
                            [ 55, 398],
                            [ 49, 397],
                            [ 45, 394],
                            [ 43, 390],
                            [ 41, 383],
                            [ 43, 236],
                            [ 44, 230],
                            [ 45, 227],
                            [141,  96],
                            [144,  93],
                            [148,  90],
                            [311,  38],
                            [315,  38]]), array([[461, 347],
                            [465, 350],
                            [468, 354],
                            [470, 360],
                            [470, 425],
                            [469, 429],
                            [467, 433],
                            [462, 437],
                            [456, 439],
                            [169, 439],
                            [165, 439],
                            [160, 436],
                            [157, 432],
                            [155, 426],
                            [154, 360],
                            [155, 356],
                            [158, 352],
                            [161, 348],
                            [168, 346],
                            [456, 346]]), array([[439, 445],
                            [441, 447],
                            [443, 451],
                            [444, 453],
                            [444, 497],
                            [443, 502],
                            [440, 504],
                            [437, 506],
                            [434, 507],
                            [189, 505],
                            [184, 504],
                            [182, 502],
                            [180, 498],
                            [179, 496],
                            [181, 453],
                            [182, 449],
                            [184, 446],
                            [188, 444],
                            [434, 444]]), array([[158, 468],
                            [199, 502],
                            [242, 522],
                            [299, 534],
                            [339, 532],
                            [373, 526],
                            [417, 508],
                            [459, 475],
                            [462, 474],
                            [467, 474],
                            [472, 476],
                            [502, 507],
                            [503, 510],
                            [504, 515],
                            [503, 518],
                            [501, 521],
                            [452, 559],
                            [450, 560],
                            [391, 584],
                            [390, 584],
                            [372, 590],
                            [370, 590],
                            [305, 596],
                            [302, 596],
                            [224, 581],
                            [221, 580],
                            [164, 553],
                            [162, 551],
                            [114, 509],
                            [112, 507],
                            [111, 503],
                            [112, 498],
                            [114, 496],
                            [146, 468],
                            [149, 466],
                            [154, 466]])], 'rec_boxes': array([], dtype=float64)}]}}
                    ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png"></p>

    === "Doc Preprocess"

        ```bash
        paddlex --pipeline doc_preprocessor \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg \
        --use_doc_orientation_classify True \
        --use_doc_unwarping True \
        --save_path ./output \
        --device gpu:0
        ```
        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'doc_test_rotated.jpg', 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg"></p>

!!! example "Computer Vision Pipelines CLI"

    === "Image Classification"

        ```bash
        paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([296, 170, 356, 258, 248], dtype=int32), 'scores': array([0.62736, 0.03752, 0.03256, 0.0323 , 0.03194], dtype=float32), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png"></p>

    === "Object Detection"

        ```bash
        paddlex --pipeline object_detection \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png \
        --threshold 0.5 \
        --save_path ./output/ \
        --device gpu:0
        ```
        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'general_object_detection_002.png', 'page_index': None, 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188614249229431, 'coordinate': [661.3518, 93.05823, 870.75903, 305.93713]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7745078206062317, 'coordinate': [76.80911, 274.74905, 330.5422, 520.0428]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7271787524223328, 'coordinate': [285.32645, 94.3175, 469.73645, 297.40344]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5576589703559875, 'coordinate': [310.8041, 361.43625, 685.1869, 712.59155]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5490103363990784, 'coordinate': [764.6252, 285.76096, 924.8153, 440.92892]}, {'cls_id': 47, 'label': 'apple', 'score': 0.515821635723114, 'coordinate': [853.9831, 169.41423, 987.803, 303.58615]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.514293372631073, 'coordinate': [0.53089714, 0.32445717, 1072.9534, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.510750949382782, 'coordinate': [57.368027, 23.455347, 213.39601, 176.45612]}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/03.png"></p>

    === "Instance Segmentation"

        ```bash
        paddlex --pipeline instance_segmentation \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png \
        --threshold 0.5 \
        --save_path ./output \
        --device gpu:0 \
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'general_instance_segmentation_004.png', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'person', 'score': 0.8695873022079468, 'coordinate': [339.83426, 0, 639.8651, 575.22003]}, {'cls_id': 0, 'label': 'person', 'score': 0.8572642803192139, 'coordinate': [0.09976959, 0, 195.07274, 575.358]}, {'cls_id': 0, 'label': 'person', 'score': 0.8201770186424255, 'coordinate': [88.24664, 113.422424, 401.23077, 574.70197]}, {'cls_id': 0, 'label': 'person', 'score': 0.7110118269920349, 'coordinate': [522.54065, 21.457964, 767.5007, 574.2464]}, {'cls_id': 27, 'label': 'tie', 'score': 0.5543721914291382, 'coordinate': [247.38776, 312.4094, 355.2685, 574.1264]}], 'masks': '...'}}
                ```

            === "img"

               <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/03.png"></p>

    === "Semantic Segmentation"

        ```bash
        paddlex --pipeline semantic_segmentation \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png \
        --target_size -1 \
        --save_path ./output \
        --device gpu:0 \
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'makassaridn-road_demo.png', 'page_index': None, 'pred': '...'}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/03.png"></p>

    === "Image Multi-label Classification"

        ```bash
        paddlex --pipeline image_multilabel_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'test_imgs/general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([21]), 'scores': array([0.99962]), 'label_names': ['bear']}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/02.png"></p>

    === "Small Object Detection"

        ```bash
        paddlex --pipeline small_object_detection \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg \
        --threshold 0.5 \
        --save_path ./output \
        --device gpu:0 \
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'small_object_detection.jpg', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'pedestrian', 'score': 0.8182944655418396, 'coordinate': [203.60147, 701.3809, 224.2007, 743.8429]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8150849342346191, 'coordinate': [185.01398, 710.8665, 201.76335, 744.9308]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748839259147644, 'coordinate': [295.1978, 500.2161, 309.33438, 532.0253]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688254714012146, 'coordinate': [851.5233, 436.13293, 863.2146, 466.8981]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.689735472202301, 'coordinate': [802.1584, 460.10693, 815.6586, 488.85086]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6697502136230469, 'coordinate': [479.947, 309.43323, 489.1534, 332.5485]}, ...]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png"></p>

    === "Image Anomaly Detection"

        ```bash
        paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0  --save_path ./output
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'input_path': 'uad_grid.png', 'pred': '...'}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_anomaly_detection/02.png"></p>

    === "3D Multi-modal Fusion Detection"

        ```bash
        paddlex --pipeline 3d_bev_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/data/nuscenes_demo.tar --device gpu:0
        ```

    === "Human Keypoint Detection"

        ```bash
        paddlex --pipeline human_keypoint_detection \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_001.jpg \
        --det_threshold 0.5 \
        --save_path ./output/ \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                    {'res': {'input_path': 'keypoint_detection_001.jpg', 'boxes': [{'coordinate': [325.65088, 74.46718, 391.5512, 209.46529], 'det_score': 0.9316536784172058, 'keypoints': array([[351.6419    ,  84.80058   ,   0.79337054],
                        [353.9377    ,  82.47209   ,   0.7778817 ],
                        [349.12946   ,  83.09801   ,   0.7885327 ],
                        [359.24466   ,  83.369225  ,   0.80503   ],
                        [347.46167   ,  84.1535    ,   0.8710606 ],
                        [368.82172   , 101.33514   ,   0.88625187],
                        [339.8064    ,  99.65537   ,   0.8432633 ],
                        [371.2092    , 123.35563   ,   0.7728337 ],
                        [337.78214   , 121.36371   ,   0.9310819 ],
                        [368.81366   , 142.71593   ,   0.79723483],
                        [337.53455   , 139.85892   ,   0.877297  ],
                        [363.0265    , 141.82988   ,   0.7964988 ],
                        [345.3075    , 141.98972   ,   0.7532031 ],
                        [374.60806   , 171.42578   ,   0.7530604 ],
                        [339.11694   , 167.98814   ,   0.7255032 ],
                        [382.67047   , 197.82553   ,   0.73685765],
                        [336.79745   , 196.5194    ,   0.626142  ]], dtype=float32), 'kpt_score': 0.7961825}, {'coordinate': [271.96713, 69.02892, 336.77832, 217.54662], 'det_score': 0.9304604530334473, 'keypoints': array([[294.48553   ,  84.144104  ,   0.74851245],
                        [297.09854   ,  80.97825   ,   0.7341483 ],
                        [292.39313   ,  81.7721    ,   0.74603605],
                        [302.3231    ,  81.528275  ,   0.7586238 ],
                        [290.6292    ,  83.26544   ,   0.7514231 ],
                        [313.32928   ,  98.40588   ,   0.83778954],
                        [286.23532   , 101.702194  ,   0.91927457],
                        [321.99515   , 120.05991   ,   0.90197486],
                        [282.39294   , 122.16547   ,   0.74502975],
                        [327.164     , 141.25995   ,   0.8172762 ],
                        [279.1632    , 133.16023   ,   0.59161717],
                        [311.02557   , 142.31526   ,   0.82111686],
                        [294.72357   , 143.42067   ,   0.71559554],
                        [313.98828   , 174.17151   ,   0.7495116 ],
                        [291.76605   , 174.39961   ,   0.7645517 ],
                        [321.4924    , 202.4499    ,   0.7817023 ],
                        [293.70663   , 204.9227    ,   0.72405976]], dtype=float32), 'kpt_score': 0.77107316}, {'coordinate': [293.55933, 188.65804, 419.47382, 305.4712], 'det_score': 0.9179267883300781, 'keypoints': array([[3.3565637e+02, 2.0941801e+02, 8.1438643e-01],
                        [3.3636591e+02, 2.0724442e+02, 7.7529407e-01],
                        [3.3486487e+02, 2.0653752e+02, 8.3719862e-01],
                        [3.4387805e+02, 2.0405179e+02, 7.9793924e-01],
                        [3.4104437e+02, 2.0354083e+02, 6.7090714e-01],
                        [3.5167136e+02, 2.1253050e+02, 5.9533423e-01],
                        [3.5493774e+02, 2.1316977e+02, 5.1632988e-01],
                        [3.2814764e+02, 2.1943013e+02, 5.3697169e-01],
                        [3.2577945e+02, 2.2027420e+02, 1.6555195e-01],
                        [3.1541614e+02, 2.2199020e+02, 5.2568728e-01],
                        [3.1139435e+02, 2.2925937e+02, 2.2075935e-01],
                        [3.8441351e+02, 2.4341478e+02, 6.4083064e-01],
                        [3.8714008e+02, 2.4532764e+02, 6.4894527e-01],
                        [3.5143246e+02, 2.5615021e+02, 7.7424920e-01],
                        [3.7133820e+02, 2.7552402e+02, 5.8704698e-01],
                        [3.6274625e+02, 2.8303183e+02, 6.1670756e-01],
                        [4.0358893e+02, 2.9351334e+02, 4.2383862e-01]], dtype=float32), 'kpt_score': 0.5969399}, {'coordinate': [238.98825, 182.67476, 372.81628, 307.61395], 'det_score': 0.914400041103363, 'keypoints': array([[282.9012    , 208.31485   ,   0.6685285 ],
                        [282.95908   , 204.36131   ,   0.66104335],
                        [280.90683   , 204.54018   ,   0.7281709 ],
                        [274.7831    , 204.04141   ,   0.54541856],
                        [270.97324   , 203.04889   ,   0.73486483],
                        [269.43472   , 217.63014   ,   0.6707946 ],
                        [256.871     , 216.546     ,   0.89603853],
                        [277.03226   , 238.2196    ,   0.4412233 ],
                        [262.29578   , 241.33434   ,   0.791063  ],
                        [292.90753   , 251.69914   ,   0.4993091 ],
                        [285.6907    , 252.71925   ,   0.7215052 ],
                        [279.36578   , 261.8949    ,   0.6626504 ],
                        [270.43402   , 268.07068   ,   0.80625033],
                        [311.96924   , 261.36716   ,   0.67315185],
                        [309.32407   , 262.97354   ,   0.72746485],
                        [345.22446   , 285.02255   ,   0.60142016],
                        [334.69235   , 291.57108   ,   0.7674925 ]], dtype=float32), 'kpt_score': 0.6821406}, {'coordinate': [66.23172, 93.531204, 124.48463, 217.99655], 'det_score': 0.9086756110191345, 'keypoints': array([[ 91.04524   , 108.79487   ,   0.8234256 ],
                        [ 92.67917   , 106.63517   ,   0.79848343],
                        [ 88.41122   , 106.8017    ,   0.8122996 ],
                        [ 95.353096  , 106.96488   ,   0.85210425],
                        [ 84.35098   , 107.85205   ,   0.971826  ],
                        [ 99.92103   , 119.87272   ,   0.853371  ],
                        [ 79.69138   , 121.08684   ,   0.8854925 ],
                        [103.019554  , 135.00996   ,   0.73513967],
                        [ 72.38997   , 136.8782    ,   0.7727014 ],
                        [104.561935  , 146.01869   ,   0.8377464 ],
                        [ 72.70636   , 151.44576   ,   0.67577386],
                        [ 98.69484   , 151.30742   ,   0.8381225 ],
                        [ 85.946     , 152.07056   ,   0.7904873 ],
                        [106.64397   , 175.77159   ,   0.8179414 ],
                        [ 84.6963    , 178.4353    ,   0.8094256 ],
                        [111.30463   , 201.2306    ,   0.74394226],
                        [ 80.08708   , 204.05814   ,   0.8457697 ]], dtype=float32), 'kpt_score': 0.8155325}, {'coordinate': [160.1294, 78.35935, 212.01868, 153.2241], 'det_score': 0.8295672535896301, 'keypoints': array([[1.89240387e+02, 9.08055573e+01, 7.36447990e-01],
                        [1.91318649e+02, 8.84640198e+01, 7.86390483e-01],
                        [1.87943207e+02, 8.88532104e+01, 8.23230743e-01],
                        [1.95832245e+02, 8.76751480e+01, 6.76276207e-01],
                        [1.86741409e+02, 8.96744080e+01, 7.87400603e-01],
                        [2.04019852e+02, 9.83068924e+01, 7.34004617e-01],
                        [1.85355087e+02, 9.81262970e+01, 6.23330474e-01],
                        [2.01501678e+02, 1.12709480e+02, 2.93740422e-01],
                        [1.80446320e+02, 1.11967369e+02, 5.50001860e-01],
                        [1.95137482e+02, 9.73322601e+01, 4.24658984e-01],
                        [1.74287552e+02, 1.21760696e+02, 3.51236403e-01],
                        [1.97997589e+02, 1.24219963e+02, 3.45360219e-01],
                        [1.83250824e+02, 1.22610085e+02, 4.38733459e-01],
                        [1.96233871e+02, 1.22864418e+02, 5.36903977e-01],
                        [1.66795364e+02, 1.25634903e+02, 3.78726840e-01],
                        [1.80727753e+02, 1.42604034e+02, 2.78717279e-01],
                        [1.75880920e+02, 1.41181213e+02, 1.70833692e-01]], dtype=float32), 'kpt_score': 0.5256467}, {'coordinate': [52.482475, 59.36664, 96.47121, 135.45993], 'det_score': 0.7726763486862183, 'keypoints': array([[ 73.98227   ,  74.01257   ,   0.71940714],
                        [ 75.44208   ,  71.73432   ,   0.6955297 ],
                        [ 72.20365   ,  71.9637    ,   0.6138198 ],
                        [ 77.7856    ,  71.665825  ,   0.73568064],
                        [ 69.342285  ,  72.25549   ,   0.6311799 ],
                        [ 83.1019    ,  77.65522   ,   0.7037722 ],
                        [ 64.89729   ,  78.846565  ,   0.56623787],
                        [ 85.16928   ,  88.88764   ,   0.5665537 ],
                        [ 61.65655   ,  89.35312   ,   0.4463089 ],
                        [ 80.01986   ,  91.51777   ,   0.30305162],
                        [ 70.90767   ,  89.90153   ,   0.48063472],
                        [ 78.70658   ,  97.33488   ,   0.39359188],
                        [ 68.3219    ,  97.67902   ,   0.41903985],
                        [ 80.69448   , 109.193985  ,   0.14496553],
                        [ 65.57641   , 105.08109   ,   0.27744702],
                        [ 79.44859   , 122.69015   ,   0.17710638],
                        [ 64.03736   , 120.170425  ,   0.46565098]], dtype=float32), 'kpt_score': 0.4905869}, {'coordinate': [7.081953, 80.3705, 46.81927, 161.72012], 'det_score': 0.6587498784065247, 'keypoints': array([[ 29.51531   ,  91.49908   ,   0.75517464],
                        [ 31.225754  ,  89.82169   ,   0.7765606 ],
                        [ 27.376017  ,  89.71614   ,   0.80448   ],
                        [ 33.515877  ,  90.82257   ,   0.7093001 ],
                        [ 23.521307  ,  90.84212   ,   0.777707  ],
                        [ 37.539314  , 101.381516  ,   0.6913692 ],
                        [ 18.340288  , 102.41546   ,   0.7203535 ],
                        [ 39.826218  , 113.37301   ,   0.5913918 ],
                        [ 16.857304  , 115.10882   ,   0.5492331 ],
                        [ 28.826103  , 121.861855  ,   0.39205936],
                        [ 22.47133   , 120.69003   ,   0.6120081 ],
                        [ 34.177963  , 126.15756   ,   0.5601723 ],
                        [ 21.39047   , 125.30078   ,   0.5064371 ],
                        [ 27.961575  , 133.33154   ,   0.54826814],
                        [ 22.303364  , 129.8608    ,   0.2293001 ],
                        [ 31.242027  , 153.047     ,   0.36292207],
                        [ 21.80127   , 153.78947   ,   0.30531448]], dtype=float32), 'kpt_score': 0.58188534}, {'coordinate': [126.131096, 30.263107, 168.5759, 134.09885], 'det_score': 0.6441988348960876, 'keypoints': array([[149.89236   ,  43.87846   ,   0.75441885],
                        [151.99484   ,  41.95912   ,   0.82070917],
                        [148.18002   ,  41.775055  ,   0.8453321 ],
                        [155.37967   ,  42.06968   ,   0.83349544],
                        [145.38167   ,  41.69159   ,   0.8233239 ],
                        [159.26329   ,  53.284737  ,   0.86246717],
                        [142.35178   ,  51.206886  ,   0.6940705 ],
                        [157.3975    ,  71.31917   ,   0.7624757 ],
                        [136.59795   ,  66.40522   ,   0.55612797],
                        [142.90988   ,  78.28269   ,   0.779243  ],
                        [135.43607   ,  73.9765    ,   0.5737738 ],
                        [155.7851    ,  82.44225   ,   0.6966109 ],
                        [143.4588    ,  80.91763   ,   0.60589534],
                        [153.45274   , 102.84818   ,   0.62720954],
                        [131.59738   ,  87.54947   ,   0.4976839 ],
                        [155.56401   , 125.58888   ,   0.5414401 ],
                        [139.57607   , 122.08866   ,   0.26570275]], dtype=float32), 'kpt_score': 0.67882234}, {'coordinate': [112.50212, 64.127, 150.35353, 125.85529], 'det_score': 0.5013833045959473, 'keypoints': array([[1.35197662e+02, 7.29378281e+01, 5.58694899e-01],
                        [1.36285202e+02, 7.16439133e+01, 6.38598502e-01],
                        [1.33776855e+02, 7.16437454e+01, 6.36756659e-01],
                        [1.37833389e+02, 7.24015121e+01, 4.13749218e-01],
                        [1.31340057e+02, 7.30362549e+01, 5.70683837e-01],
                        [1.42542435e+02, 8.28875885e+01, 2.30803847e-01],
                        [1.29773300e+02, 8.52729874e+01, 4.94463116e-01],
                        [1.41332916e+02, 9.43963928e+01, 9.36751068e-02],
                        [1.28858521e+02, 9.95147858e+01, 2.72373617e-01],
                        [1.44981277e+02, 7.83604965e+01, 8.68032947e-02],
                        [1.34379593e+02, 8.23366165e+01, 1.67876005e-01],
                        [1.37895874e+02, 1.08476562e+02, 1.58305198e-01],
                        [1.30837265e+02, 1.07525513e+02, 1.45044222e-01],
                        [1.31290604e+02, 1.02961494e+02, 7.68775940e-02],
                        [1.17951675e+02, 1.07433502e+02, 2.09531561e-01],
                        [1.29175934e+02, 1.14402641e+02, 1.46551579e-01],
                        [1.27901909e+02, 1.16773926e+02, 2.08665460e-01]], dtype=float32), 'kpt_score': 0.3005561}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg"></p>

    === "Open Vocabulary Segmentation"

        ```bash
        paddlex --pipeline open_vocabulary_segmentation \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg \
        --prompt_type box \
        --prompt "[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]" \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'open_vocabulary_segmentation.jpg', 'prompts': {'box_prompt': [[112.9, 118.4, 513.8, 382.1], [4.6, 263.6, 92.2, 336.6], [592.4, 260.9, 607.2, 294.2]]}, 'masks': '...', 'mask_infos': [{'label': 'box_prompt', 'prompt': [112.9, 118.4, 513.8, 382.1]}, {'label': 'box_prompt', 'prompt': [4.6, 263.6, 92.2, 336.6]}, {'label': 'box_prompt', 'prompt': [592.4, 260.9, 607.2, 294.2]}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg"></p>

    === "Open Vocabulary Detection"

        ```bash
        paddlex --pipeline open_vocabulary_detection \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg \
        --prompt "bus . walking man . rearview mirror ." \
        --thresholds "{'text_threshold': 0.25, 'box_threshold': 0.3}" \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'open_vocabulary_detection.jpg', 'page_index': None, 'boxes': [{'coordinate': [112.10542297363281, 117.93667602539062, 514.35693359375, 382.10150146484375], 'label': 'bus', 'score': 0.9348853230476379}, {'coordinate': [264.1828918457031, 162.6674346923828, 286.8844909667969, 201.86187744140625], 'label': 'rearview mirror', 'score': 0.6022508144378662}, {'coordinate': [606.1133422851562, 254.4973907470703, 622.56982421875, 293.7867126464844], 'label': 'walking man', 'score': 0.4384709894657135}, {'coordinate': [591.8192138671875, 260.2451171875, 607.3953247070312, 294.2210388183594], 'label': 'man', 'score': 0.3573091924190521}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_detection/open_vocabulary_detection_res.jpg"></p>

    === "Pedestrian Attribute Recognition"

        ```bash
        paddlex --pipeline pedestrian_attribute_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'pedestrian_attribute_002.jpg', 'boxes': [{'labels': ['Trousers(长裤)', 'Age18-60(年龄在18-60岁之间)', 'LongCoat(长外套)', 'Side(侧面)'], 'cls_scores': array([0.99965, 0.99963, 0.98866, 0.9624 ]), 'det_score': 0.9795178771018982, 'coordinate': [87.24581, 322.5872, 546.2697, 1039.9852]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Front(面朝前)', 'Age18-60(年龄在18-60岁之间)'], 'cls_scores': array([0.99996, 0.99872, 0.93379, 0.71614]), 'det_score': 0.967143177986145, 'coordinate': [737.91626, 306.287, 1150.5961, 1034.2979]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Age18-60(年龄在18-60岁之间)', 'Side(侧面)'], 'cls_scores': array([0.99996, 0.99514, 0.98726, 0.96224]), 'det_score': 0.9645745754241943, 'coordinate': [399.45944, 281.9107, 869.5312, 1038.9962]}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg"></p>

    === "Vehicle Attribute Recognition"

        ```bash
        paddlex --pipeline vehicle_attribute_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg --device gpu:0
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'vehicle_attribute_002.jpg', 'boxes': [{'labels': ['red(红色)', 'sedan(轿车)'], 'cls_scores': array([0.96375, 0.94025]), 'det_score': 0.9774094820022583, 'coordinate': [196.32553, 302.3847, 639.3131, 655.57904]}, {'labels': ['suv(SUV)', 'brown(棕色)'], 'cls_scores': array([0.99968, 0.99317]), 'det_score': 0.9705657958984375, 'coordinate': [769.4419, 278.8417, 1401.0217, 641.3569]}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/vehicle_attribute_recognition/01.jpg"></p>

    === "Rotated Object Detection"

        ```bash
        paddlex --pipeline rotated_object_detection \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png \
        --threshold 0.5 \
        --save_path ./output \
        --device gpu:0 \
        ```

        ??? question "What's the result"
            === "output"
                ```bash
                {'res': {'input_path': 'rotated_object_detection_001.png', 'page_index': None, 'boxes': [{'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7409099340438843, 'coordinate': [92.88687, 763.1569, 85.163124, 749.5868, 116.07975, 731.99414, 123.8035, 745.5643]}, {'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7393015623092651, 'coordinate': [348.2332, 177.55974, 332.77704, 150.24973, 345.2183, 143.21028, 360.67444, 170.5203]}, {'cls_id': 11, 'label': 'roundabout', 'score': 0.8101699948310852, 'coordinate': [537.1732, 695.5475, 204.4297, 612.9735, 286.71338, 281.48022, 619.4569, 364.05426]}]}}
                ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/rotated_object_detection_001_res.png"></p>

!!! example "Time Series-related CLI"

    === "Time Series Forecasting"

        ```bash
        paddlex --pipeline ts_forecast --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --save_path ./output --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
                {'input_path': 'ts_fc.csv', 'forecast':                            OT
                date
                2018-06-26 20:00:00  9.586131
                2018-06-26 21:00:00  9.379762
                2018-06-26 22:00:00  9.252275
                2018-06-26 23:00:00  9.249993
                2018-06-27 00:00:00  9.164998
                ...                       ...
                2018-06-30 15:00:00  8.830340
                2018-06-30 16:00:00  9.291553
                2018-06-30 17:00:00  9.097666
                2018-06-30 18:00:00  8.905430
                2018-06-30 19:00:00  8.993793

                [96 rows x 1 columns]}
            ```

    === "Time Series Anomaly Detection"

        ```bash
        paddlex --pipeline ts_anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv --device gpu:0 --save_path ./output
        ```

        ??? question "What's the result"
            ```bash
                {'input_path': 'ts_ad.csv', 'anomaly':            label
                timestamp
                220226         0
                220227         0
                220228         0
                220229         0
                220230         0
                ...          ...
                220317         1
                220318         1
                220319         1
                220320         1
                220321         0

                [96 rows x 1 columns]}
            ```

    === "Time Series Classification"

        ```bash
        paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
                {'input_path': 'ts_cls.csv', 'classification':         classid     score
                sample
                0             0  0.617688}
            ```

!!! example "Speech-related Pipelines CLI"

    === "Multilingual Speech Recognition"

        ```bash
        paddlex --pipeline multilingual_speech_recognition \
        --input https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav \
        --save_path ./output \
        --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
                {'input_path': 'zh.wav', 'result': {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.0, 'text': '我认为跑步最重要的就是', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 50464, 50464, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50564], 'temperature': 0, 'avg_logprob': -0.22779104113578796, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.026114309206604958}, {'id': 1, 'seek': 200, 'start': 2.0, 'end': 31.0, 'text': '给我带来了身体健康', 'tokens': [50364, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 51814], 'temperature': 0, 'avg_logprob': -0.21976988017559052, 'compression_ratio': 0.23684210526315788, 'no_speech_prob': 0.009023111313581467}], 'language': 'zh'}}
            ```


!!! example "Video-related Pipelines CLI"

    === "Video Classification"

        ```bash
        paddlex --pipeline video_classification \
            --input https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4 \
            --topk 5 \
            --save_path ./output \
            --device gpu:0
        ```

        ??? question "What's the result"
            ```bash
            {'res': {'input_path': 'general_video_classification_001.mp4', 'class_ids': array([  0, 278,  68, 272, 162], dtype=int32), 'scores': [0.91996, 0.07055, 0.00235, 0.00215, 0.00158], 'label_names': ['abseiling', 'rock_climbing', 'climbing_tree', 'riding_mule', 'ice_climbing']}}
            ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/02.jpg"></p>

    === "Video Detection"

        ```bash
        paddlex --pipeline video_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi --device gpu:0 --save_path output
        ```

        ??? question "What's the result"
            ```bash
            {'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121, 8, 192, 195], 0.8197548738023599, 'HorseRiding']]]}
            ```

            === "img"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_detection/HorseRiding_res.jpg"></p>

## 📝 Python Usage

A few lines of code can complete the quick inference of the production line, with a unified Python script format as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[production line name])
output = pipeline.predict([input image name])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
The following steps were executed:

* `create_pipeline()` instantiates the production line object
* Pass in the image and call the `predict` method of the production line object for inference prediction
* Process the prediction results


!!! example "OCR-related Python"

    === "OCR"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="OCR")

        output = pipeline.predict(
            input="./general_ocr_002.png",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "Table Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline_name="table_recognition")

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

    === "Table Recognition v2"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline_name="table_recognition_v2")

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

    === "General Layout Parsing"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="layout_parsing")

        output = pipeline.predict(
            input="./demo_paper.png",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img(save_path="./output/") ## Save the result in img format
            res.save_to_json(save_path="./output/") ## Save the result in json format
            res.save_to_xlsx(save_path="./output/") ## Save the result in table format
            res.save_to_html(save_path="./output/") ## Save the result in html format
        ```

    === "Formula Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="formula_recognition")

        output = pipeline.predict(
            input="./general_formula_recognition_001.png",
            use_layout_detection=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            layout_threshold=0.5,
            layout_nms=True,
            layout_unclip_ratio=1.0,
            layout_merge_bboxes_mode="large"
        )
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "Seal Text Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="seal_recognition")

        output = pipeline.predict(
            "seal_text_det.png",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img("./output/") ## Save the visualization result
            res.save_to_json("./output/") ## Save the visualization result
        ```

    === "Doc Preprocessor"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="doc_preprocessor")
        output = pipeline.predict(
            input="doc_test_rotated.jpg",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
        )
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

!!! example "Computer Vision Pipeline Command-Line Usage"

    === "General Image Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="image_classification")

        output = pipeline.predict("general_image_classification_001.jpg")
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img(save_path="./output/") ## Save the visualized result image
            res.save_to_json(save_path="./output/") ## Save the structured output of the prediction
        ```

    === "General Object Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="object_detection")

        output = pipeline.predict("general_object_detection_002.png", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "General Instance Segmentation"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="instance_segmentation")
        output = pipeline.predict(input="general_instance_segmentation_004.png", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "General Semantic Segmentation"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="semantic_segmentation")
        output = pipeline.predict(input="general_semantic_segmentation_002.png", target_size = -1)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "Image Multi-label Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="image_multilabel_classification")

        output = pipeline.predict("general_image_classification_001.jpg")
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img("./output/") ## Save the visualized result image
            res.save_to_json("./output/") ## Save the structured output of the prediction
        ```

    === "Small Object Detection"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="small_object_detection")
        output = pipeline.predict(input="small_object_detection.jpg", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "Image Anomaly Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="anomaly_detection")
        output = pipeline.predict(input="uad_grid.png")
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img(save_path="./output/") ## Save the visualized result image
            res.save_to_json(save_path="./output/") ## Save the structured output of the prediction
        ```

    === "3D Multi-modal Fusion Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="3d_bev_detection")
        output = pipeline.predict("./data/nuscenes_demo/nuscenes_infos_val.pkl")

        for res in output:
            res.print()  ## Print the structured output of the prediction
            res.save_to_json("./output/")  ## Save the result to a JSON file
        ```

    === "Human Keypoint Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="human_keypoint_detection")

        output = pipeline.predict("keypoint_detection_001.jpg", det_threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "Open-Vocabulary Segmentation"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="open_vocabulary_segmentation")
        output = pipeline.predict(input="open_vocabulary_segmentation.jpg", prompt_type="box", prompt=[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]])
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "Open-Vocabulary Detection"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="open_vocabulary_detection")
        output = pipeline.predict(input="open_vocabulary_detection.jpg", prompt="bus . walking man . rearview mirror .")
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "Pedestrian Attribute Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

        output = pipeline.predict("pedestrian_attribute_002.jpg")
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img("./output/") ## Save the visualized result image
            res.save_to_json("./output/") ## Save the structured output of the prediction
        ```

    === "Vehicle Attribute Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="vehicle_attribute_recognition")

        output = pipeline.predict("vehicle_attribute_002.jpg")
        for res in output:
            res.print() ## Print the structured output of the prediction
            res.save_to_img("./output/") ## Save the visualized result image
            res.save_to_json("./output/") ## Save the structured output of the prediction
        ```

    === "Rotated Object Detection"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="rotated_object_detection")
        output = pipeline.predict(input="rotated_object_detection_001.png", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

!!! example "Command Line Usage for Time Series Production Lines"

    === "Time Series Forecasting"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_forecast")

        output = pipeline.predict(input="ts_fc.csv")
        for res in output:
            res.print() ## Print the structured prediction output
            res.save_to_csv(save_path="./output/") ## Save results in CSV format
            res.save_to_json(save_path="./output/") ## Save results in JSON format
        ```

    === "Time Series Anomaly Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_anomaly_detection")
        output = pipeline.predict("ts_ad.csv")
        for res in output:
            res.print() ## Print the structured prediction output
            res.save_to_csv(save_path="./output/") ## Save results in CSV format
            res.save_to_json(save_path="./output/") ## Save results in JSON format
        ```

    === "Time Series Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_cls")
        output = pipeline.predict("ts_cls.csv")
        for res in output:
            res.print() ## Print the structured prediction output
            res.save_to_csv(save_path="./output/") ## Save results in CSV format
            res.save_to_json(save_path="./output/") ## Save results in JSON format
        ```

!!! example "Command Line Usage for Speech Production Lines"

    === "Multilingual Speech Recognition"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="multilingual_speech_recognition")
        output = pipeline.predict(input="zh.wav")

        for res in output:
            res.print()
            res.save_to_json(save_path="./output/")
        ```

!!! example "Command Line Usage for Video Production Lines"

    === "General Video Classification"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="video_classification")

        output = pipeline.predict("general_video_classification_001.mp4", topk=5)
        for res in output:
            res.print()
            res.save_to_video(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "General Video Detection"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="video_detection")
        output = pipeline.predict(input="HorseRiding.avi")
        for res in output:
            res.print() ## Print the structured prediction output
            res.save_to_video(save_path="./output/") ## Save the visualized video results
            res.save_to_json(save_path="./output/") ## Save the structured prediction output
        ```

## 🚀 Detailed Tutorials

<div class="grid cards" markdown>

- **Document Information Extraction**

    ---

    Document scene information extraction v3 (PP-ChatOCRv3-doc) is a document and image intelligent analysis solution with PaddlePaddle features, combining LLM and OCR technologies to solve complex document information extraction challenges such as layout analysis, rare character recognition, multi-page PDF, table, and seal recognition in one stop.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.en.md)

- **OCR**

    ---

    The general OCR production line is used to solve text recognition tasks, extract text information from images, and output it in text form. Based on the end-to-end OCR system, it can achieve millisecond-level precise text content prediction on CPUs and reach open-source SOTA in general scenarios.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/ocr_pipelines/OCR.en.md)

- **Image Classification**

    ---

    Image classification can automatically extract image features and classify them accurately, recognizing various objects such as animals, plants, traffic signs, etc., and is widely used in object recognition, scene understanding, and automatic tagging fields.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/cv_pipelines/image_classification.en.md)

- **Object Detection**

    ---

    Object detection aims to identify the categories and locations of multiple objects in images or videos by generating bounding boxes to mark these objects. This technology is widely used in fields such as autonomous driving, surveillance systems, and smart photo albums.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/cv_pipelines/object_detection.en.md)

- **Small Object Detection**

    ---

    Small object detection is a technology specifically designed to recognize smaller objects in images, widely used in surveillance, unmanned driving, and satellite image analysis fields. It can accurately locate and classify small-sized objects such as pedestrians, traffic signs, or small animals from complex scenes.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/cv_pipelines/small_object_detection.en.md)

- **Time Series Forecasting**

    ---

    Time series forecasting is a technique that uses historical data to predict future trends by analyzing the patterns of change in time series data. It is widely used in financial markets, weather forecasting, and sales forecasting fields.

    [:octicons-arrow-right-24: Tutorial](pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.en.md)

</div>

[:octicons-arrow-right-24: More](pipeline_usage/pipeline_develop_guide.en.md)


## 💬 Discussion

We warmly welcome and encourage community members to raise questions, share ideas, and feedback in the [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) section. Whether you want to report a bug, discuss a feature request, seek help, or just want to keep up with the latest project news, this is a great platform.
