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
    <a href=""><img src="https://img.shields.io/badge/Python-3.8~3.12-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows%2C%20Mac-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>


## 🔍 简介

PaddleX 3.0 是基于飞桨框架构建的低代码开发工具，它集成了众多<b>开箱即用的预训练模型</b>，可以实现模型从训练到推理的<b>全流程开发</b>，支持国内外<b>多款主流硬件</b>，助力AI 开发者进行产业实践。

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

![PaddleX](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/PaddleX_ch.png)


## 🛠️ 安装

!!! warning
    在安装 PaddleX 之前，请确保您已具备基本的 <b>Python 运行环境</b>（注：目前支持 <b>Python 3.8 至 Python 3.12</b>）。PaddleX 3.0-rc1 版本依赖的 PaddlePaddle 版本为 <b>3.0.0以上版本</b>。

### 安装 PaddlePaddle

=== "CPU 版本"
    ```bash
    python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    ```
=== "GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）"
    ```bash
     python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ```
=== "GPU 版本，需显卡驱动程序版本 ≥550.54.14（Linux）或 ≥550.54.14（Windows）"
    ```bash
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    ```

!!!tip
    无需关注物理机上的 CUDA 版本，只需关注显卡驱动程序版本。更多飞桨 Wheel 版本信息，请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html)。



### 安装PaddleX

```bash
pip install "paddlex[base]"
```

> ❗ 更多安装方式参考 [PaddleX 安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)

## 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

PaddleX的每一条产线对应特定的参数，您可以在各自的产线文档中查看具体的参数说明。每条产线需指定必要的三个参数：

* `pipeline`：产线名称或产线配置文件
* `input`：待处理的输入文件（如图片）的本地路径、目录或 URL
* `device`: 使用的硬件设备及序号（例如`gpu:0`表示使用第 0 块 GPU），也可选择使用 NPU(`npu:0`)、 XPU(`xpu:0`)、CPU(`cpu`)等

!!! example "OCR相关产线命令行使用"

    === "通用OCR"

        ```bash
        paddlex --pipeline OCR \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
                --use_doc_orientation_classify False \
                --use_doc_unwarping False \
                --use_textline_orientation False \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': False}, 'angle': 0},'dt_polys': [array([[ 3, 10],[82, 10],[82, 33],[ 3, 33]], dtype=int16), ...], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, ...], 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.99*', ...], 'rec_scores': [0.8980069160461426,  ...], 'rec_polys': [array([[ 3, 10],[82, 10],[82, 33],[ 3, 33]], dtype=int16), ...], 'rec_boxes': array([[  3,  10,  82,  33], ...], dtype=int16)}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/03.png"></p>

    === "通用表格识别"

        ```bash
        paddlex --pipeline table_recognition \
                --use_doc_orientation_classify=False \
                --use_doc_unwarping=False \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': '/root/.paddlex/predict_input/table_recognition.jpg', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 8, 'label': 'table', 'score': 0.9730289578437805, 'coordinate': [0.77032924, 0.0992564, 550.78864, 127.53717]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[234,   6],
                        ...,
                        [234,  25]],

                    ...,

                    [[448, 101],
                        ...,
                        [448, 121]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': array([0.99612021, ..., 0.99815977]), 'rec_polys': array([[[234,   6],
                        ...,
                        [234,  25]],

                    ...,

                    [[448, 101],
                        ...,
                        [448, 121]]], dtype=int16), 'rec_boxes': array([[234, ...,  25],
                    ...,
                    [448, ..., 121]], dtype=int16)}, 'table_res_list': [{'cell_box_list': array([[  3.77032924, ...,  27.0992564 ],
                    ...,
                    [403.77032924, ..., 125.0992564 ]]), 'pred_html': '<html><body><table><tr><td colspan="4">CRuncover</td></tr><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上 没想</td><td>江、整江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>', 'table_ocr_pred': {'rec_polys': array([[[234,   6],
                        ...,
                        [234,  25]],

                    ...,

                    [[448, 101],
                        ...,
                        [448, 121]]], dtype=int16), 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': array([0.99612021, ..., 0.99815977]), 'rec_boxes': array([[234, ...,  25],
                    ...,
                    [448, ..., 121]], dtype=int16)}}]}}
                ```

            === "可视化图片"
                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png"></p>

    === "通用表格识别v2"

        ```bash
        paddlex --pipeline table_recognition_v2 \
                --use_doc_orientation_classify=False \
                --use_doc_unwarping=False \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition_v2.jpg \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'table_recognition_v2.jpg', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 8, 'label': 'table', 'score': 0.86655592918396, 'coordinate': [0.0125130415, 0.41920784, 1281.3737, 585.3884]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[   9,   21],
                        ...,
                        [   9,   59]],

                    ...,

                    [[1046,  536],
                        ...,
                        [1046,  573]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['部门', '报销人', '报销事由', '批准人：', '单据', '张', '合计金额', '元', '车费票', '其', '火车费票', '飞机票', '中', '旅住宿费', '其他', '补贴'], 'rec_scores': array([0.99958128, ..., 0.99317062]), 'rec_polys': array([[[   9,   21],
                        ...,
                        [   9,   59]],

                    ...,

                    [[1046,  536],
                        ...,
                        [1046,  573]]], dtype=int16), 'rec_boxes': array([[   9, ...,   59],
                    ...,
                    [1046, ...,  573]], dtype=int16)}, 'table_res_list': [{'cell_box_list': [array([ 0.13052222, ..., 73.08310249]), array([104.43082511, ...,  73.27777413]), array([319.39041221, ...,  73.30439308]), array([424.2436837 , ...,  73.44736794]), array([580.75836265, ...,  73.24003914]), array([723.04370201, ...,  73.22717598]), array([984.67315757, ...,  73.20420387]), array([1.25130415e-02, ..., 5.85419208e+02]), array([984.37072837, ..., 137.02281502]), array([984.26586998, ..., 201.22290352]), array([984.24017417, ..., 585.30775765]), array([1039.90606773, ...,  265.44664314]), array([1039.69549644, ...,  329.30540779]), array([1039.66546714, ...,  393.57319954]), array([1039.5122689 , ...,  457.74644783]), array([1039.55535972, ...,  521.73030403]), array([1039.58612144, ...,  585.09468392])], 'pred_html': '<html><body><table><tbody><tr><td>部门</td><td></td><td>报销人</td><td></td><td>报销事由</td><td></td><td colspan="2">批准人：</td></tr><tr><td colspan="6" rowspan="8"></td><td colspan="2">单据 张</td></tr><tr><td colspan="2">合计金额 元</td></tr><tr><td rowspan="6">其 中</td><td>车费票</td></tr><tr><td>火车费票</td></tr><tr><td>飞机票</td></tr><tr><td>旅住宿费</td></tr><tr><td>其他</td></tr><tr><td>补贴</td></tr></tbody></table></body></html>', 'table_ocr_pred': {'rec_polys': array([[[   9,   21],
                        ...,
                        [   9,   59]],

                    ...,

                    [[1046,  536],
                        ...,
                        [1046,  573]]], dtype=int16), 'rec_texts': ['部门', '报销人', '报销事由', '批准人：', '单据', '张', '合计金额', '元', '车费票', '其', '火车费票', '飞机票', '中', '旅住宿费', '其他', '补贴'], 'rec_scores': array([0.99958128, ..., 0.99317062]), 'rec_boxes': array([[   9, ...,   59],
                    ...,
                    [1046, ...,  573]], dtype=int16)}}]}}
                ```

            === "可视化图片"
                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/02.jpg"></p>


    === "通用版面解析"

        ```bash
        paddlex --pipeline layout_parsing \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_parsing_demo.png \
                --use_doc_orientation_classify False \
                --use_doc_unwarping False \
                --use_textline_orientation False \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
            ```bash
            {'res': {'input_path': 'layout_parsing_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': False}, 'parsing_res_list': [{'block_bbox': [133.36868, 40.128025, 1383.7496, 123.51852], 'block_label': 'text', 'block_content': '助力双方交往\n搭建友谊桥梁'}, {'block_bbox': [587.4096, 160.58267, 927.6319, 179.2817], 'block_label': 'figure_title', 'block_content': '本报记者沈小晓任彦黄培昭'}, {'block_bbox': [773.8337, 200.6484, 1505.5646, 687.1228], 'block_label': 'image', 'block_content': ''}, {'block_bbox': [390.39642, 201.85085, 741.43414, 292.60092], 'block_label': 'text', 'block_content': '厄立特里亚高等教育与研究院合作建立，开\n设了中国语言课程和中国文化课程，注册学\n生2万余人次。10余年来，厄特孔院已成为\n当地民众了解中国的一扇窗口。'}, {'block_bbox': [9.714512, 202.68811, 359.62323, 340.3127], 'block_label': 'text', 'block_content': '身着中国传统民族服装的厄立特里亚青\n年依次登台表演中国民族舞、现代舞、扇子舞\n等，曼妙的舞姿赢得现场观众阵阵掌声。这\n是日前厄立特里亚高等教育与研究院孔子学\n院(以下简称"厄特孔院"举办“喜迎新年"中国\n歌舞比赛的场景。'}, {'block_bbox': [390.74124, 298.42255, 740.8009, 436.79193], 'block_label': 'text', 'block_content': '黄鸣飞表示，随着来学习中文的人日益\n增多，阿斯马拉大学教学点已难以满足教学\n需要。2024年4月，由中企蜀道集团所属四\n川路桥承建的孔院教学楼项目在阿斯马拉开\n工建设，预计今年上半年峻工，建成后将为厄\n特孔院提供全新的办学场地。'}, {'block_bbox': [10.579921, 346.26508, 359.13733, 436.17682], 'block_label': 'text', 'block_content': '中国和厄立特里亚传统友谊深厚。近年\n来,在高质量共建“一带一路"框架下，中厄两\n国人文交流不断深化，互利合作的民意基础\n日益深厚。'}, {'block_bbox': [410.51892, 457.0816, 722.768, 516.78217], 'block_label': 'text', 'block_content': '“在中国学习的经历\n让我看到更广阔的世界”'}, {'block_bbox': [30.334734, 457.53757, 341.92758, 516.81995], 'block_label': 'paragraph_title', 'block_content': '“学好中文，我们的\n未来不是梦"'}, {'block_bbox': [390.89084, 538.1777, 742.1934, 604.6777], 'block_label': 'text', 'block_content': '多年来，厄立特里亚广大赴华留学生和\n培训人员积极投身国家建设，成为助力该国\n发展的人才和厄中友好的见证者和推动者。'}, {'block_bbox': [9.884378, 538.3683, 359.43878, 652.03644], 'block_label': 'text', 'block_content': '“鲜花曾告诉我你怎样走过，大地知道你\n心中的每一个角落……"厄立特里亚阿斯马拉\n大学综合楼二层，一阵优美的歌声在走廊里回\n响。循着熟悉的旋律轻轻推开一间教室的门，\n学生们正跟着老师学唱中文歌曲《同一首歌》。'}, {'block_bbox': [390.88583, 610.61304, 741.1856, 747.91656], 'block_label': 'text', 'block_content': '在厄立特里亚全国妇女联盟工作的约翰\n娜·特韦尔德·凯莱塔就是其中一位。她曾在\n中华女子学院攻读硕士学位，研究方向是女\n性领导力与社会发展。其间，她实地走访中国\n多个地区，获得了观察中国社会发展的第一\n手资料。'}, {'block_bbox': [10.115321, 658.7913, 359.40344, 771.3188], 'block_label': 'text', 'block_content': '这是厄特孔院阿斯马拉大学教学点的一\n节中文歌曲课。为了让学生们更好地理解歌\n词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐\n字翻译和解释歌词。随着伴奏声响起，学生们\n边唱边随着节拍摇动身体，现场气氛热烈。'}, {'block_bbox': [809.6597, 705.4076, 1485.5686, 747.42346], 'block_label': 'figure_title', 'block_content': '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。\n中国驻厄立特里亚大使馆供图'}, {'block_bbox': [389.62894, 753.4464, 742.0593, 890.9599], 'block_label': 'text', 'block_content': '谈起在中国求学的经历，约翰娜记忆犹\n新：“中国的发展在当今世界是独一无二的。\n沿着中国特色社会主义道路坚定前行，中国\n创造了发展奇迹，这一切都离不开中国共产党\n的领导。中国的发展经验值得许多国家学习\n借鉴。”'}, {'block_bbox': [9.867948, 777.38995, 360.40143, 843.43], 'block_label': 'text', 'block_content': '“这是中文歌曲初级班，共有32人。学\n生大部分来自首都阿斯马拉的中小学，年龄\n最小的仅有6岁。"尤斯拉告诉记者。'}, {'block_bbox': [9.780596, 850.09344, 359.62875, 1059.8483], 'block_label': 'text', 'block_content': '尤斯拉今年23岁，是厄立特里亚一所公立\n学校的艺术老师。她12岁开始在厄特孔院学\n习中文，在2017年第十届"汉语桥"世界中学生\n中文比赛中获得厄立特里亚赛区第一名，并和\n同伴代表厄立特里亚前往中国参加决赛，获得\n团体优胜奖。2022年起，尤斯拉开始在厄特孔\n院兼职教授中文歌曲，每周末两个课时。“中国\n文化博大精深，我希望我的学生们能够通过中\n文歌曲更好地理解中国文化。"她说。'}, {'block_bbox': [771.98157, 777.02783, 1124.4025, 1059.2194], 'block_label': 'text', 'block_content': '“不管远近都是客人，请不用客气；相约\n好了在一起，我们欢迎你…"在一场中厄青\n年联谊活动上，四川路桥中方员工同当地大\n学生合唱《北京欢迎你》。厄立特里亚技术学\n院计算机科学与工程专业学生鲁夫塔·谢拉\n是其中一名演唱者，她很早便在孔院学习中\n文，一直在为去中国留学作准备。“这句歌词\n是我们两国人民友谊的生动写照。无论是投\n身于厄立特里亚基础设施建设的中企员工，\n还是在中国留学的厄立特里亚学子，两国人\n民携手努力，必将推动两国关系不断向前发\n展。"鲁夫塔说。'}, {'block_bbox': [1155.9126, 777.7057, 1331.4768, 795.6466], 'block_label': 'text', 'block_content': '瓦的北红海省博物馆。'}, {'block_bbox': [1153.6993, 801.5608, 1504.5693, 987.6245], 'block_label': 'text', 'block_content': '博物馆二层陈列着一个发掘自阿杜利\n斯古城的中国古代陶制酒器，罐身上写着\n“万”“和"“禅"“山"等汉字。“这件文物证\n明，很早以前我们就通过海上丝绸之路进行\n贸易往来与文化交流。这也是厄立特里亚\n与中国友好交往历史的有力证明。"北红海\n省博物馆研究与文献部负责人伊萨亚斯·特\n斯法兹吉说。'}, {'block_bbox': [390.19556, 897.58984, 742.03076, 1035.8021], 'block_label': 'text', 'block_content': '正在西南大学学习的厄立特里亚博士生\n穆卢盖塔·泽穆伊对中国怀有深厚感情。8\n盖塔在社交媒体上写下这样一段话：“这是我\n人生的重要一步，自此我拥有了一双坚固的\n鞋子，赋予我穿越荆棘的力量。"'}, {'block_bbox': [1154.4485, 993.47705, 1503.8442, 1107.7343], 'block_label': 'text', 'block_content': '厄立特里亚国家博物馆考古学和人类学\n研究员菲尔蒙·特韦尔德十分喜爱中国文\n化。他表示：“学习彼此的语言和文化，将帮\n助厄中两国人民更好地理解彼此，助力双方\n交往，搭建友谊桥梁。"'}, {'block_bbox': [391.1788, 1041.2518, 740.86304, 1131.4594], 'block_label': 'text', 'block_content': '穆卢盖塔密切关注中国在经济、科技、教\n育等领域的发展，“中国在科研等方面的实力\n与日俱增。在中国学习的经历让我看到更广\n阔的世界，从中受益匪浅。”'}, {'block_bbox': [9.44888, 1065.2853, 360.20395, 1180.0414], 'block_label': 'text', 'block_content': '“姐姐，你想去中国吗？"“非常想！我想\n去看故宫、爬长城。"尤斯拉的学生中有一对\n能歌善舞的姐妹，姐姐露娅今年15岁，妹妹\n莉娅14岁，两人都已在厄特孔院学习多年，\n中文说得格外流利。'}, {'block_bbox': [771.5103, 1065.0969, 1123.4521, 1179.5637], 'block_label': 'text', 'block_content': '厄立特里亚高等教育委员会主任助理萨\n马瑞表示：“每年我们都会组织学生到中国访\n问学习，目前有超过5000名厄立特里亚学生\n在中国留学。学习中国的教育经验，有助于\n提升厄立特里亚的教育水平。"'}, {'block_bbox': [1153.9072, 1114.012, 1503.9468, 1347.0771], 'block_label': 'text', 'block_content': '厄立特里亚国家博物馆馆长塔吉丁·努\n里达姆·优素福曾多次访问中国，对中华文明\n的传承与创新、现代化博物馆的建设与发展\n印象深刻。“中国博物馆不仅有许多保存完好\n的文物，还充分运用先进科技手段进行展示，\n帮助人们更好理解中华文明。"塔吉丁说，“厄\n立特里亚与中国都拥有悠久的文明，始终相\n互理解、相互尊重。我希望未来与中国同行\n加强合作，共同向世界展示非洲和亚洲的灿\n烂文明。”'}, {'block_bbox': [390.84042, 1137.481, 741.0446, 1346.7771], 'block_label': 'text', 'block_content': '23岁的莉迪亚·埃斯蒂法诺斯已在厄特\n孔院学习3年，在中国书法、中国画等方面表\n现十分优秀，在2024年厄立特里亚赛区的\n“汉语桥"比赛中获得一等奖。莉迪亚说：“学\n习中国书法让我的内心变得安宁和纯粹。我\n也喜欢中国的服饰，希望未来能去中国学习，\n把中国不同民族元素融入服装设计中，创作\n出更多精美作品，也把厄特文化分享给更多\n的中国朋友。”'}, {'block_bbox': [8.636964, 1186.1119, 359.81888, 1299.478], 'block_label': 'text', 'block_content': '露娅对记者说：“这些年来，怀着对中文\n和中国文化的热爱，我们姐妹俩始终相互鼓\n励，一起学习。我们的中文一天比一天好，还\n学会了中文歌和中国舞。我们一定要到中国\n去。学好中文，我们的未来不是梦！”'}, {'block_bbox': [9.65557, 1305.0817, 359.62732, 1347.9409], 'block_label': 'text', 'block_content': '据厄特孔院中方院长黄鸣飞介绍，这所\n孔院成立于2013年3月，由贵州财经大学和'}, {'block_bbox': [791.91876, 1201.0499, 1104.4822, 1260.1809], 'block_label': 'text', 'block_content': '“共同向世界展示非\n洲和亚洲的灿烂文明”'}, {'block_bbox': [772.4803, 1281.0001, 1123.4064, 1348.0051], 'block_label': 'text', 'block_content': '从阿斯马拉出发，沿着蜿蜓曲折的盘山\n公路一路向东寻找丝路印迹。驱车两个小\n时，记者来到位于厄立特里亚港口城市马萨'}], 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 1, 'label': 'image', 'score': 0.985334038734436, 'coordinate': [773.8337, 200.6484, 1505.5646, 687.1228]}, {'cls_id': 2, 'label': 'text', 'score': 0.9781306385993958, 'coordinate': [771.98157, 777.02783, 1124.4025, 1059.2194]}, {'cls_id': 2, 'label': 'text', 'score': 0.9772591590881348, 'coordinate': [1153.9072, 1114.012, 1503.9468, 1347.0771]}, {'cls_id': 2, 'label': 'text', 'score': 0.9764459133148193, 'coordinate': [390.74124, 298.42255, 740.8009, 436.79193]}, {'cls_id': 2, 'label': 'text', 'score': 0.9752683639526367, 'coordinate': [9.714512, 202.68811, 359.62323, 340.3127]}, {'cls_id': 2, 'label': 'text', 'score': 0.9751600027084351, 'coordinate': [1153.6993, 801.5608, 1504.5693, 987.6245]}, {'cls_id': 2, 'label': 'text', 'score': 0.9741775989532471, 'coordinate': [9.780596, 850.09344, 359.62875, 1059.8483]}, {'cls_id': 2, 'label': 'text', 'score': 0.9723023772239685, 'coordinate': [390.39642, 201.85085, 741.43414, 292.60092]}, {'cls_id': 2, 'label': 'text', 'score': 0.9717830419540405, 'coordinate': [390.84042, 1137.481, 741.0446, 1346.7771]}, {'cls_id': 2, 'label': 'text', 'score': 0.970496654510498, 'coordinate': [390.88583, 610.61304, 741.1856, 747.91656]}, {'cls_id': 2, 'label': 'text', 'score': 0.967951774597168, 'coordinate': [8.636964, 1186.1119, 359.81888, 1299.478]}, {'cls_id': 2, 'label': 'text', 'score': 0.9675101637840271, 'coordinate': [390.19556, 897.58984, 742.03076, 1035.8021]}, {'cls_id': 2, 'label': 'text', 'score': 0.9672409296035767, 'coordinate': [389.62894, 753.4464, 742.0593, 890.9599]}, {'cls_id': 2, 'label': 'text', 'score': 0.9657630324363708, 'coordinate': [10.579921, 346.26508, 359.13733, 436.17682]}, {'cls_id': 2, 'label': 'text', 'score': 0.9655840396881104, 'coordinate': [771.5103, 1065.0969, 1123.4521, 1179.5637]}, {'cls_id': 2, 'label': 'text', 'score': 0.9651231169700623, 'coordinate': [1154.4485, 993.47705, 1503.8442, 1107.7343]}, {'cls_id': 2, 'label': 'text', 'score': 0.9631907343864441, 'coordinate': [772.4803, 1281.0001, 1123.4064, 1348.0051]}, {'cls_id': 2, 'label': 'text', 'score': 0.9616568684577942, 'coordinate': [9.44888, 1065.2853, 360.20395, 1180.0414]}, {'cls_id': 2, 'label': 'text', 'score': 0.9596402645111084, 'coordinate': [10.115321, 658.7913, 359.40344, 771.3188]}, {'cls_id': 2, 'label': 'text', 'score': 0.9591165781021118, 'coordinate': [391.1788, 1041.2518, 740.86304, 1131.4594]}, {'cls_id': 2, 'label': 'text', 'score': 0.9564457535743713, 'coordinate': [9.884378, 538.3683, 359.43878, 652.03644]}, {'cls_id': 2, 'label': 'text', 'score': 0.9525545835494995, 'coordinate': [390.89084, 538.1777, 742.1934, 604.6777]}, {'cls_id': 2, 'label': 'text', 'score': 0.9493281245231628, 'coordinate': [9.867948, 777.38995, 360.40143, 843.43]}, {'cls_id': 2, 'label': 'text', 'score': 0.9400925040245056, 'coordinate': [9.65557, 1305.0817, 359.62732, 1347.9409]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9255779385566711, 'coordinate': [809.6597, 705.4076, 1485.5686, 747.42346]}, {'cls_id': 2, 'label': 'text', 'score': 0.9046083688735962, 'coordinate': [1155.9126, 777.7057, 1331.4768, 795.6466]}, {'cls_id': 2, 'label': 'text', 'score': 0.8666954040527344, 'coordinate': [410.51892, 457.0816, 722.768, 516.78217]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.7959974408149719, 'coordinate': [30.334734, 457.53757, 341.92758, 516.81995]}, {'cls_id': 2, 'label': 'text', 'score': 0.7298153042793274, 'coordinate': [791.91876, 1201.0499, 1104.4822, 1260.1809]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6074362397193909, 'coordinate': [587.4096, 160.58267, 927.6319, 179.2817]}, {'cls_id': 2, 'label': 'text', 'score': 0.584621012210846, 'coordinate': [133.36868, 40.128025, 1383.7496, 123.51852]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[ 122,   28],
                    ...,
                    [ 122,  135]],

                ...,

                [[1156, 1330],
                    ...,
                    [1156, 1351]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '西', '本报记者沈小晓任彦黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等，曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院"举办“喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建“一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年峻工，建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦"', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落……"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新：“中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起，我们欢迎你………"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。"尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和”“禅”“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明，很早以前我们就通过海上丝绸之路进行', '习中文，在2017年第十届"汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话：“这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。"', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？"“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。"', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。”', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。"', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说：“这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '印象深刻。“中国博物馆不仅有许多保存完好', '“共同向世界展示非', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说：“学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，“厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰，希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！”', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜓曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99982357, ..., 0.93637466]), 'rec_polys': array([[[ 122,   28],
                    ...,
                    [ 122,  135]],

                ...,

                [[1156, 1330],
                    ...,
                    [1156, 1351]]], dtype=int16), 'rec_boxes': array([[ 122, ...,  135],
                ...,
                [1156, ..., 1351]], dtype=int16)}}}
            ```
    === "通用版面解析v3"

        ```bash
        paddlex --pipeline PP-StructureV3 \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png \
                --use_doc_orientation_classify False \
                --use_doc_unwarping False \
                --use_textline_orientation False \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"

            ```bash
            {'res': {'input_path': 'pp_structure_v3_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9853514432907104, 'coordinate': [770.9531, 776.6814, 1122.6057, 1058.7322]}, {'cls_id': 1, 'label': 'image', 'score': 0.9848673939704895, 'coordinate': [775.7434, 202.27979, 1502.8113, 686.02136]}, {'cls_id': 2, 'label': 'text', 'score': 0.983731746673584, 'coordinate': [1152.3197, 1113.3275, 1503.3029, 1346.586]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832221865653992, 'coordinate': [1152.5602, 801.431, 1503.8436, 986.3563]}, {'cls_id': 2, 'label': 'text', 'score': 0.9829439520835876, 'coordinate': [9.549545, 849.5713, 359.1173, 1058.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9811657667160034, 'coordinate': [389.58298, 1137.2659, 740.66235, 1346.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9775941371917725, 'coordinate': [9.1302185, 201.85, 359.0409, 339.05692]}, {'cls_id': 2, 'label': 'text', 'score': 0.9750366806983948, 'coordinate': [389.71454, 752.96924, 740.544, 889.92456]}, {'cls_id': 2, 'label': 'text', 'score': 0.9738152027130127, 'coordinate': [389.94565, 298.55988, 740.5585, 435.5124]}, {'cls_id': 2, 'label': 'text', 'score': 0.9737328290939331, 'coordinate': [771.50256, 1065.4697, 1122.2582, 1178.7324]}, {'cls_id': 2, 'label': 'text', 'score': 0.9728517532348633, 'coordinate': [1152.5154, 993.3312, 1503.2349, 1106.327]}, {'cls_id': 2, 'label': 'text', 'score': 0.9725610017776489, 'coordinate': [9.372787, 1185.823, 359.31738, 1298.7227]}, {'cls_id': 2, 'label': 'text', 'score': 0.9724331498146057, 'coordinate': [389.62848, 610.7389, 740.83234, 746.2377]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720287322998047, 'coordinate': [389.29898, 897.0936, 741.41516, 1034.6616]}, {'cls_id': 2, 'label': 'text', 'score': 0.9713053703308105, 'coordinate': [10.323685, 1065.4663, 359.6786, 1178.8872]}, {'cls_id': 2, 'label': 'text', 'score': 0.9689728021621704, 'coordinate': [9.336395, 537.6609, 359.2901, 652.1881]}, {'cls_id': 2, 'label': 'text', 'score': 0.9684857130050659, 'coordinate': [10.7608185, 345.95068, 358.93616, 434.64087]}, {'cls_id': 2, 'label': 'text', 'score': 0.9681928753852844, 'coordinate': [9.674866, 658.89075, 359.56528, 770.4319]}, {'cls_id': 2, 'label': 'text', 'score': 0.9634978175163269, 'coordinate': [770.9464, 1281.1785, 1122.6522, 1346.7156]}, {'cls_id': 2, 'label': 'text', 'score': 0.96304851770401, 'coordinate': [390.0113, 201.28055, 740.1684, 291.53073]}, {'cls_id': 2, 'label': 'text', 'score': 0.962053120136261, 'coordinate': [391.21393, 1040.952, 740.5046, 1130.32]}, {'cls_id': 2, 'label': 'text', 'score': 0.9565253853797913, 'coordinate': [10.113251, 777.1482, 359.439, 842.437]}, {'cls_id': 2, 'label': 'text', 'score': 0.9497362375259399, 'coordinate': [390.31357, 537.86285, 740.47595, 603.9285]}, {'cls_id': 2, 'label': 'text', 'score': 0.9371236562728882, 'coordinate': [10.2034, 1305.9753, 359.5958, 1346.7295]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9338151216506958, 'coordinate': [791.6062, 1200.8479, 1103.3257, 1259.9324]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9326773285865784, 'coordinate': [408.0737, 457.37024, 718.9509, 516.63464]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9274250864982605, 'coordinate': [29.448685, 456.6762, 340.99194, 515.6999]}, {'cls_id': 2, 'label': 'text', 'score': 0.8742568492889404, 'coordinate': [1154.7095, 777.3624, 1330.3086, 794.5853]}, {'cls_id': 2, 'label': 'text', 'score': 0.8442489504814148, 'coordinate': [586.49316, 160.15454, 927.468, 179.64203]}, {'cls_id': 11, 'label': 'doc_title', 'score': 0.8332607746124268, 'coordinate': [133.80017, 37.41908, 1380.8601, 124.1429]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6770150661468506, 'coordinate': [812.1718, 705.1199, 1484.6973, 747.1692]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[133,  35],
                    ...,
                    [133, 131]],

                ...,

                [[ 13, 754],
                    ...,
                    [ 13, 777]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者', '沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等,曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院")举办"喜迎新年"中国', '黄鸣飞表示,随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设,预计今年上半年竣工,建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '多年来,厄立特里亚广大赴华留学生和', '培训人员积极投身国家建设,成为助力该国', '发展的人才和厄中友好的见证者和推动者。', '在厄立特里亚全国妇女联盟工作的约翰', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '中华女子学院攻读硕士学位,研究方向是女', '性领导力与社会发展。其间，她实地走访中国', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '手资料。', '中国驻厄立特里亚大使馆供图', '“这是中文歌曲初级班，共有32人。学', '“不管远近都是客人，请不用客气;相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '好了在一起,我们欢迎你"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。"尤斯拉告诉记者。', '年联谊活动上,四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器,罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和""禅”“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明,很早以前我们就通过海上丝绸之路进行', '习中文,在2017年第十届"汉语桥"世界中学生', '是其中一名演唱者,她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名,并和', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛,获得', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲,每周末两个课时。中国', '还是在中国留学的厄立特里亚学子,两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深,我希望我的学生们能够通过中', '民携手努力,必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐,你想去中国吗?"“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往,搭建友谊桥梁。"', '能歌善舞的姐妹,姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验,有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说：“这些年来,怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '“共同向世界展示非', '印象深刻。“中国博物馆不仅有许多保存完好', '和中国文化的热爱,我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说：“学', '的文物,还充分运用先进科技手段进行展示，', '励,一起学习。我们的中文一天比一天好,还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰,希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明,始终相', '去。学好中文,我们的未来不是梦!"', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发,沿着蜿蜒曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍,这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作,共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时,记者来到位于厄立特里亚港口城市马萨', '烂文明。”', '谈起在中国求学的经历,约翰娜记忆犹', '新：“中国的发展在当今世界是独一无二的。', '沿着中国特色社会主义道路坚定前行，中国', '创造了发展奇迹,这一切都离不开中国共产党', '的领导。中国的发展经验值得许多国家学习', '借鉴，”', '正在西南大学学习的厄立特里亚博士生', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '年前，在北京师范大学获得硕士学位后，穆卢', '盖塔在社交媒体上写下这样一段话：“这是我', '人生的重要一步，自此我拥有了一双坚固的', '鞋子.赋予我穿越荆棘的力量。”', '“鲜花曾告诉我你怎样走过，大地知道你', '心中的每一个角落"厄立特里亚阿斯马拉', '大学综合楼二层，一阵优美的歌声在走廊里回', '响。循着熟悉的旋律轻轻推开一间教室的门，', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '这是厄特孔院阿斯马拉大学教学点的一', '节中文歌曲课。为了让学生们更好地理解歌', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '字翻译和解释歌词。随着伴奏声响起，学生们', '边唱边随着节拍摇动身体，现场气氛热烈。'], 'rec_scores': array([0.99972075, ..., 0.96241361]), 'rec_polys': array([[[133,  35],
                    ...,
                    [133, 131]],

                ...,

                [[ 13, 754],
                    ...,
                    [ 13, 777]]], dtype=int16), 'rec_boxes': array([[133, ..., 131],
                ...,
                [ 13, ..., 777]], dtype=int16)}}}
            ```

    === "公式识别"

        ```bash
        paddlex --pipeline formula_recognition \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png \
                --use_layout_detection True \
                --use_doc_orientation_classify False \
                --use_doc_unwarping False \
                --layout_threshold 0.5 \
                --layout_nms True \
                --layout_unclip_ratio  1.0 \
                --layout_merge_bboxes_mode "'large'"\
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                    {'res': {'input_path': 'general_formula_recognition.png', 'model_settings': {'use_doc_preprocessor': False,'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9778407216072083, 'coordinate': [271.257, 648.50824, 1040.2291, 774.8482]}, ...]}, 'formula_res_list': [{'rec_formula': '\\small\\begin{aligned}{p(\\mathbf{x})=c(\\mathbf{u})\\prod_{i}p(x_{i}).}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([553.0718, 802.0996, 758.75635, 853.093],)}, ...]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04.png"></p>

    === "印章文本识别"

        ```bash
        paddlex --pipeline seal_recognition \
            --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png \
            --use_doc_orientation_classify False \
            --use_doc_unwarping False \
            --device gpu:0 \
            --save_path ./output
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'seal_text_det.png', 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 16, 'label': 'seal', 'score': 0.975531280040741, 'coordinate': [6.195526, 0.1579895, 634.3982, 628.84595]}]}, 'seal_res_list': [{'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[320,  38],
                    ...,
                    [315,  38]]), array([[461, 347],
                    ...,
                    [456, 346]]), array([[439, 445],
                    ...,
                    [434, 444]]), array([[158, 468],
                    ...,
                    [154, 466]])], 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.2, 'box_thresh': 0.6, 'unclip_ratio': 0.5}, 'text_type': 'seal', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['天津君和缘商贸有限公司', '发票专用章', '吗繁物', '5263647368706'], 'rec_scores': array([0.9934051 , ..., 0.99139398]), 'rec_polys': [array([[320,  38],
                    ...,
                    [315,  38]]), array([[461, 347],
                    ...,
                    [456, 346]]), array([[439, 445],
                    ...,
                    [434, 444]]), array([[158, 468],
                    ...,
                    [154, 466]])], 'rec_boxes': array([], dtype=float64)}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png"></p>

    === "文档图像预处理"

        ```bash
        paddlex --pipeline doc_preprocessor \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg \
                --use_doc_orientation_classify True \
                --use_doc_unwarping True \
                --save_path ./output \
                --device gpu:0
        ```
        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'doc_test_rotated.jpg', 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg"></p>


!!! example "计算机视觉相关产线命令行使用"

    === "通用图像分类"

        ```bash
        paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([296, 170, 356, 258, 248], dtype=int32), 'scores': array([0.62736, 0.03752, 0.03256, 0.0323 , 0.03194], dtype=float32), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png"></p>

    === "通用目标检测"

        ```bash
        paddlex --pipeline object_detection \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png \
                --threshold 0.5 \
                --save_path ./output/ \
                --device gpu:0
        ```
        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'general_object_detection_002.png', 'page_index': None, 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188614249229431, 'coordinate': [661.3518, 93.05823, 870.75903, 305.93713]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7745078206062317, 'coordinate': [76.80911, 274.74905, 330.5422, 520.0428]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7271787524223328, 'coordinate': [285.32645, 94.3175, 469.73645, 297.40344]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5576589703559875, 'coordinate': [310.8041, 361.43625, 685.1869, 712.59155]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5490103363990784, 'coordinate': [764.6252, 285.76096, 924.8153, 440.92892]}, {'cls_id': 47, 'label': 'apple', 'score': 0.515821635723114, 'coordinate': [853.9831, 169.41423, 987.803, 303.58615]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.514293372631073, 'coordinate': [0.53089714, 0.32445717, 1072.9534, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.510750949382782, 'coordinate': [57.368027, 23.455347, 213.39601, 176.45612]}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/object_detection/03.png"></p>

    === "通用实例分割"

        ```bash
        paddlex --pipeline instance_segmentation \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png \
                --threshold 0.5 \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'general_instance_segmentation_004.png', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'person', 'score': 0.8695873022079468, 'coordinate': [339.83426, 0, 639.8651, 575.22003]}, {'cls_id': 0, 'label': 'person', 'score': 0.8572642803192139, 'coordinate': [0.09976959, 0, 195.07274, 575.358]}, {'cls_id': 0, 'label': 'person', 'score': 0.8201770186424255, 'coordinate': [88.24664, 113.422424, 401.23077, 574.70197]}, {'cls_id': 0, 'label': 'person', 'score': 0.7110118269920349, 'coordinate': [522.54065, 21.457964, 767.5007, 574.2464]}, {'cls_id': 27, 'label': 'tie', 'score': 0.5543721914291382, 'coordinate': [247.38776, 312.4094, 355.2685, 574.1264]}], 'masks': '...'}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/instance_segmentation/03.png"></p>

    === "通用语义分割"

        ```bash
        paddlex --pipeline semantic_segmentation \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png \
                --target_size -1 \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'makassaridn-road_demo.png', 'page_index': None, 'pred': '...'}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/03.png"></p>

    === "图像多标签分类"

        ```bash
        paddlex --pipeline image_multilabel_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'test_imgs/general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([21]), 'scores': array([0.99962]), 'label_names': ['bear']}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/02.png"></p>

    === "小目标检测"

        ```bash
        paddlex --pipeline small_object_detection \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg \
                --threshold 0.5 \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'small_object_detection.jpg', 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'pedestrian', 'score': 0.8182944655418396, 'coordinate': [203.60147, 701.3809, 224.2007, 743.8429]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8150849342346191, 'coordinate': [185.01398, 710.8665, 201.76335, 744.9308]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748839259147644, 'coordinate': [295.1978, 500.2161, 309.33438, 532.0253]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688254714012146, 'coordinate': [851.5233, 436.13293, 863.2146, 466.8981]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.689735472202301, 'coordinate': [802.1584, 460.10693, 815.6586, 488.85086]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6697502136230469, 'coordinate': [479.947, 309.43323, 489.1534, 332.5485]}, ...]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png"></p>

    === "图像异常检测"

        ```bash
        paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0  --save_path ./output
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'input_path': 'uad_grid.png', 'pred': '...'}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_anomaly_detection/02.png"></p>

    === "3D多模态融合检测"

        ```bash
        paddlex --pipeline 3d_bev_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/data/nuscenes_demo.tar --device gpu:0
        ```

    === "人体关键点检测"

        ```bash
        paddlex --pipeline human_keypoint_detection \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_001.jpg \
                --det_threshold 0.5 \
                --save_path ./output/ \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
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

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg"></p>

    === "开放词汇分割"

        ```bash
        paddlex --pipeline open_vocabulary_segmentation \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg \
                --prompt_type box \
                --prompt "[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]" \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'open_vocabulary_segmentation.jpg', 'prompts': {'box_prompt': [[112.9, 118.4, 513.8, 382.1], [4.6, 263.6, 92.2, 336.6], [592.4, 260.9, 607.2, 294.2]]}, 'masks': '...', 'mask_infos': [{'label': 'box_prompt', 'prompt': [112.9, 118.4, 513.8, 382.1]}, {'label': 'box_prompt', 'prompt': [4.6, 263.6, 92.2, 336.6]}, {'label': 'box_prompt', 'prompt': [592.4, 260.9, 607.2, 294.2]}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg"></p>

    === "开放词汇检测"

        ```bash
        paddlex --pipeline open_vocabulary_detection \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg \
                --prompt "bus . walking man . rearview mirror ." \
                --thresholds "{'text_threshold': 0.25, 'box_threshold': 0.3}" \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'open_vocabulary_detection.jpg', 'page_index': None, 'boxes': [{'coordinate': [112.10542297363281, 117.93667602539062, 514.35693359375, 382.10150146484375], 'label': 'bus', 'score': 0.9348853230476379}, {'coordinate': [264.1828918457031, 162.6674346923828, 286.8844909667969, 201.86187744140625], 'label': 'rearview mirror', 'score': 0.6022508144378662}, {'coordinate': [606.1133422851562, 254.4973907470703, 622.56982421875, 293.7867126464844], 'label': 'walking man', 'score': 0.4384709894657135}, {'coordinate': [591.8192138671875, 260.2451171875, 607.3953247070312, 294.2210388183594], 'label': 'man', 'score': 0.3573091924190521}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_detection/open_vocabulary_detection_res.jpg"></p>

    === "行人属性识别"

        ```bash
        paddlex --pipeline pedestrian_attribute_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'pedestrian_attribute_002.jpg', 'boxes': [{'labels': ['Trousers(长裤)', 'Age18-60(年龄在18-60岁之间)', 'LongCoat(长外套)', 'Side(侧面)'], 'cls_scores': array([0.99965, 0.99963, 0.98866, 0.9624 ]), 'det_score': 0.9795178771018982, 'coordinate': [87.24581, 322.5872, 546.2697, 1039.9852]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Front(面朝前)', 'Age18-60(年龄在18-60岁之间)'], 'cls_scores': array([0.99996, 0.99872, 0.93379, 0.71614]), 'det_score': 0.967143177986145, 'coordinate': [737.91626, 306.287, 1150.5961, 1034.2979]}, {'labels': ['Trousers(长裤)', 'LongCoat(长外套)', 'Age18-60(年龄在18-60岁之间)', 'Side(侧面)'], 'cls_scores': array([0.99996, 0.99514, 0.98726, 0.96224]), 'det_score': 0.9645745754241943, 'coordinate': [399.45944, 281.9107, 869.5312, 1038.9962]}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/pedestrian_attribute_recognition/01.jpg"></p>

    === "车辆属性识别"

        ```bash
        paddlex --pipeline vehicle_attribute_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'vehicle_attribute_002.jpg', 'boxes': [{'labels': ['red(红色)', 'sedan(轿车)'], 'cls_scores': array([0.96375, 0.94025]), 'det_score': 0.9774094820022583, 'coordinate': [196.32553, 302.3847, 639.3131, 655.57904]}, {'labels': ['suv(SUV)', 'brown(棕色)'], 'cls_scores': array([0.99968, 0.99317]), 'det_score': 0.9705657958984375, 'coordinate': [769.4419, 278.8417, 1401.0217, 641.3569]}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/vehicle_attribute_recognition/01.jpg"></p>

    === "旋转目标检测"

        ```bash
        paddlex --pipeline rotated_object_detection \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png \
                --threshold 0.5 \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            === "输出结果"
                ```bash
                {'res': {'input_path': 'rotated_object_detection_001.png', 'page_index': None, 'boxes': [{'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7409099340438843, 'coordinate': [92.88687, 763.1569, 85.163124, 749.5868, 116.07975, 731.99414, 123.8035, 745.5643]}, {'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7393015623092651, 'coordinate': [348.2332, 177.55974, 332.77704, 150.24973, 345.2183, 143.21028, 360.67444, 170.5203]}, {'cls_id': 11, 'label': 'roundabout', 'score': 0.8101699948310852, 'coordinate': [537.1732, 695.5475, 204.4297, 612.9735, 286.71338, 281.48022, 619.4569, 364.05426]}]}}
                ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/rotated_object_detection_001_res.png"></p>

!!! example "时序分析相关产线命令行使用"

    === "时序预测"

        ```bash
        paddlex --pipeline ts_forecast --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0 --save_path ./output
        ```

        ??? question "查看运行结果"
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

    === "时序异常检测"

        ```bash
        paddlex --pipeline ts_anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv --device gpu:0 --save_path ./output
        ```

        ??? question "查看运行结果"
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

    === "时序分类"

        ```bash
        paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0
        ```

        ??? question "查看运行结果"
            ```bash
                {'input_path': 'ts_cls.csv', 'classification':         classid     score
                sample
                0             0  0.617688}
            ```

!!! example "语音相关产线命令行使用"

    === "多语种语音识别"

        ```bash
        paddlex --pipeline multilingual_speech_recognition \
                --input https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            ```bash
                {'input_path': 'zh.wav', 'result': {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.0, 'text': '我认为跑步最重要的就是', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 50464, 50464, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50564], 'temperature': 0, 'avg_logprob': -0.22779104113578796, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.026114309206604958}, {'id': 1, 'seek': 200, 'start': 2.0, 'end': 31.0, 'text': '给我带来了身体健康', 'tokens': [50364, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 51814], 'temperature': 0, 'avg_logprob': -0.21976988017559052, 'compression_ratio': 0.23684210526315788, 'no_speech_prob': 0.009023111313581467}], 'language': 'zh'}}
            ```


!!! example "视频相关产线命令行使用"

    === "通用视频分类"

        ```bash
        paddlex --pipeline video_classification \
                --input https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4 \
                --topk 5 \
                --save_path ./output \
                --device gpu:0
        ```

        ??? question "查看运行结果"
            ```bash
            {'res': {'input_path': 'general_video_classification_001.mp4', 'class_ids': array([  0, 278,  68, 272, 162], dtype=int32), 'scores': [0.91996, 0.07055, 0.00235, 0.00215, 0.00158], 'label_names': ['abseiling', 'rock_climbing', 'climbing_tree', 'riding_mule', 'ice_climbing']}}
            ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/02.jpg"></p>

    === "通用视频检测"

        ```bash
        paddlex --pipeline video_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi --device gpu:0 --save_path output
        ```

        ??? question "查看运行结果"
            ```bash
            {'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121, 8, 192, 195], 0.8197548738023599, 'HorseRiding']]]}
            ```

            === "可视化图片"

                <p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_detection/HorseRiding_res.jpg"></p>


## 📝 Python 脚本使用

几行代码即可完成产线的快速推理，统一的 Python 脚本格式如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[产线名称])
output = pipeline.predict([输入图片名称])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的 `predict()` 方法进行推理预测
* 对预测结果进行处理

!!! example "OCR相关产线Python脚本使用"

    === "通用OCR"

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

    === "通用表格识别"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="table_recognition")

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

    === "通用表格识别v2"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="table_recognition_v2")

        output = pipeline.predict(
            input="table_recognition_v2.jpg",
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

    === "通用版面解析"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="layout_parsing")

        output = pipeline.predict(
            input="./layout_parsing_demo.png",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img(save_path="./output/") ## 保存当前图像的所有子模块预测的可视化图像结果
            res.save_to_json(save_path="./output/") ## 保存当前图像的结构化json结果
            res.save_to_xlsx(save_path="./output/") ## 保存当前图像的子表格xlsx格式的结果
            res.save_to_html(save_path="./output/") ## 保存当前图像的子表格html格式的结果

        ```

    === "通用版面解析v3"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="PP-StructureV3")

        output = pipeline.predict(
            input="./pp_structure_v3_demo.png",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        for res in output:
            res.print()
            res.save_to_json(save_path="output")
            res.save_to_markdown(save_path="output")

        ```


    === "公式识别"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="formula_recognition")

        output = pipeline.predict(
            input="./general_formula_recognition_001.png",
            use_layout_detection=True ,
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

    === "印章文本识别"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="seal_recognition")

        output = pipeline.predict(
            "seal_text_det.png",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img("./output/") ## 保存可视化结果
            res.save_to_json("./output/") ## 保存可视化结果
        ```

    === "文档图像预处理"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline="doc_preprocessor")
        output = pipeline.predict(
            input="doc_test_rotated.jpg"
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
        )
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

!!! example "计算机视觉相关产线Python脚本使用"

    === "通用图像分类"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="image_classification")

        output = pipeline.predict("general_image_classification_001.jpg")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img(save_path="./output/") ## 保存结果可视化图像
            res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
        ```

    === "通用目标检测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="object_detection")

        output = pipeline.predict("general_object_detection_002.png", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "通用实例分割"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="instance_segmentation")
        output = pipeline.predict(input="general_instance_segmentation_004.png", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "通用语义分割"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="semantic_segmentation")
        output = pipeline.predict(input="general_semantic_segmentation_002.png", target_size=-1)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "图像多标签分类"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="image_multilabel_classification")

        output = pipeline.predict("general_image_classification_001.jpg")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img("./output/") ## 保存结果可视化图像
            res.save_to_json("./output/") ## 保存预测的结构化输出
        ```

    === "小目标检测"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="small_object_detection")
        output = pipeline.predict(input="small_object_detection.jpg", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "图像异常检测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="anomaly_detection")
        output = pipeline.predict(input="uad_grid.png")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img(save_path="./output/") ## 保存结果可视化图像
            res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
        ```

    === "3D多模态融合检测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="3d_bev_detection")
        output = pipeline.predict("./data/nuscenes_demo/nuscenes_infos_val.pkl")

        for res in output:
            res.print()  ## 打印预测的结构化输出
            res.save_to_json("./output/")  ## 保存结果到json文件
        ```

    === "人体关键点检测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="human_keypoint_detection")

        output = pipeline.predict("keypoint_detection_001.jpg"， det_threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img("./output/")
            res.save_to_json("./output/")
        ```

    === "开放词汇分割"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="open_vocabulary_segmentation")
        output = pipeline.predict(input="open_vocabulary_segmentation.jpg", prompt_type="box", prompt=[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]])
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "开放词汇检测"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="open_vocabulary_detection")
        output = pipeline.predict(input="open_vocabulary_detection.jpg", prompt="bus . walking man . rearview mirror .")
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "行人属性识别"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

        output = pipeline.predict("pedestrian_attribute_002.jpg")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img("./output/") ## 保存结果可视化图像
            res.save_to_json("./output/") ## 保存预测的结构化输出
        ```

    === "车辆属性识别"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="vehicle_attribute_recognition")

        output = pipeline.predict("vehicle_attribute_002.jpg")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_img("./output/") ## 保存结果可视化图像
            res.save_to_json("./output/") ## 保存预测的结构化输出
        ```

    === "旋转目标检测"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline_name="rotated_object_detection")
        output = pipeline.predict(input="rotated_object_detection_001.png", threshold=0.5)
        for res in output:
            res.print()
            res.save_to_img(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

!!! example "时序分析相关产线Python脚本使用"

    === "时序预测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_forecast")

        output = pipeline.predict(input="ts_fc.csv")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_csv(save_path="./output/") ## 保存csv格式结果
            res.save_to_json(save_path="./output/") ## 保存json格式结果
        ```

    === "时序异常检测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_anomaly_detection")
        output = pipeline.predict("ts_ad.csv")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_csv(save_path="./output/") ## 保存csv格式结果
            res.save_to_json(save_path="./output/") ## 保存json格式结果
        ```

    === "时序分类"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="ts_classification")
        output = pipeline.predict("ts_cls.csv")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_csv(save_path="./output/") ## 保存csv格式结果
            res.save_to_json(save_path="./output/") ## 保存json格式结果
        ```

!!! example "语音相关产线Python脚本使用"

    === "多语种语音识别"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="multilingual_speech_recognition")
        output = pipeline.predict(input="zh.wav")

        for res in output:
            res.print()
            res.save_to_json(save_path="./output/")
        ```

!!! example "视频相关产线Python脚本使用"

    === "通用视频分类"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="video_classification")

        output = pipeline.predict("general_video_classification_001.mp4", topk=5)
        for res in output:
            res.print()
            res.save_to_video(save_path="./output/")
            res.save_to_json(save_path="./output/")
        ```

    === "通用视频检测"

        ```python
        from paddlex import create_pipeline

        pipeline = create_pipeline(pipeline="video_detection")
        output = pipeline.predict(input="HorseRiding.avi")
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_video(save_path="./output/") ## 保存结果可视化视频
            res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
        ```

!!! example "多模态视觉语言模型相关产线Python脚本使用"

    === "文档理解"

        ```python
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline="doc_understanding")
        output = pipeline.predict(
            {
                "image": "medal_table.png",
                "query": "识别这份表格的内容"
            }
        )
        for res in output:
            res.print() ## 打印预测的结构化输出
            res.save_to_json("./output/") ## 保存预测的结构化输出
        ```

## 🚀 详细教程

<div class="grid cards" markdown>

- **文档信息抽取v4**

    ---

    文档场景信息抽取v4（PP-ChatOCRv4-doc）是飞桨特色的文档和图像智能分析解决方案，结合了 LLM、MLLM 和 OCR 技术，一站式解决版面分析、生僻字、多页 pdf、表格、印章识别等常见的复杂文档信息抽取难点问题。

    [:octicons-arrow-right-24: 教程](pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.md)

- **通用OCR**

    ---

    通用 OCR 产线用于解决文字识别任务，提取图片中的文字信息并以文本形式输出，基于端到端 OCR 串联系统，可实现 CPU 上毫秒级的文本内容精准预测，在通用场景上达到开源SOTA。

    [:octicons-arrow-right-24: 教程](pipeline_usage/tutorials/ocr_pipelines/OCR.md)

- **通用版面解析v3**

    ---

    通用版面解析v3产线在通用版面解析v1产线的基础上，强化了版面区域检测、表格识别、公式识别的能力，增加了多栏阅读顺序的恢复能力、结果转换 Markdown 文件的能力，在多种文档数据中，表现优异，可以处理较复杂的文档数据。

    [:octicons-arrow-right-24: 教程](pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.md)

- **通用表格识别产线v2**

    ---

    通用表格识别产线v2用于解决表格识别任务，对图片中的表格进行识别，并以HTML格式输出。基于本产线，可实现对表格的精准预测，使用场景覆盖通用、制造、金融、交通等各个领域。

    [:octicons-arrow-right-24: 教程](pipeline_usage/tutorials/cv_pipelines/table_recognition_v2.md)

- **小目标检测**

    ---

    小目标检测是一种专门识别图像中体积较小物体的技术，广泛应用于监控、无人驾驶和卫星图像分析等领域。它能够从复杂场景中准确找到并分类像行人、交通标志或小动物等小尺寸物体。

    [:octicons-arrow-right-24: 教程](pipeline_usage/tutorials/cv_pipelines/small_object_detection.md)

- **时序预测**

    ---

    时序预测是一种利用历史数据来预测未来趋势的技术，通过分析时间序列数据的变化模式。广泛应用于金融市场、天气预报和销售预测等领域。

    [:octicons-arrow-right-24: 教程](pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.md)

</div>

[:octicons-arrow-right-24: 更多](pipeline_usage/pipeline_develop_guide.md)

## 💬 Discussion

我们非常欢迎并鼓励社区成员在 [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) 板块中提出问题、分享想法和反馈。无论您是想要报告一个 bug、讨论一个功能请求、寻求帮助还是仅仅想要了解项目的最新动态，这里都是一个绝佳的平台。
