[简体中文](layout_parsing.md) | English

# General Layout Parsing Pipeline Tutorial

## 1. Introduction to the General Layout Parsing Pipeline
Layout parsing is a technology that extracts structured information from document images, primarily used to convert complex document layouts into machine-readable data formats. This technology has extensive applications in document management, information extraction, and data digitization. By combining Optical Character Recognition (OCR), image processing, and machine learning algorithms, layout parsing can identify and extract text blocks, titles, paragraphs, images, tables, and other layout elements from documents. The process typically involves three main steps: layout analysis, element analysis, and data formatting, ultimately generating structured document data to improve data processing efficiency and accuracy.

The **General Layout Parsing Pipeline** includes modules for table structure recognition, layout region analysis, text detection, text recognition, formula recognition, seal text detection, text image rectification, and document image orientation classification.

**If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.**

<details>
   <summary> 👉Model List Details</summary>

**Table Structure Recognition Module Models**:

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>SLANet</td>
    <td>59.52</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
  </tr>
  <tr>
    <td>SLANet_plus</td>
    <td>63.69</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet_plus is an enhanced version of SLANet, the table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves the recognition ability for wireless and complex tables and reduces the model's sensitivity to the accuracy of table positioning, enabling more accurate recognition even with offset table positioning.</td>
  </tr>
</table>

**Note: The above accuracy metrics are measured on PaddleX's internally built English table recognition dataset. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Layout Detection Module Models**:

| Model | mAP(0.5) (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size (M) | Description |
|-|-|-|-|-|-|
| PicoDet_layout_1x | 86.8 | 13.0 | 91.3 | 7.4 | An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists. |
|PicoDet-S_layout_3cls|87.1|13.5 |45.8 |4.8|An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals. |
|PicoDet-S_layout_17cls|70.3|13.6|46.2|4.8|A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals. |
|PicoDet-L_layout_3cls|89.3|15.7|159.8|22.6|An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals. |
|PicoDet-L_layout_17cls|79.9|17.2 |160.2|22.6|A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals. |
| RT-DETR-H_layout_3cls | 95.9 | 114.6 | 3832.6 | 470.1 | A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals. |
| RT-DETR-H_layout_17cls | 92.6 | 115.1 | 3827.2 | 470.2 | A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals. |

**Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout region analysis dataset, containing 10,000 images of common document types, including English and Chinese papers, magazines, research reports, etc. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Text Detection Module Models**:

| Model | Detection Hmean (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size (M) | Description |
|-------|---------------------|-------------------------|-------------------------|--------------|-------------|
| PP-OCRv4_server_det | 82.69 | 83.3501 | 2434.01 | 109 | PP-OCRv4's server-side text detection model, featuring higher accuracy, suitable for deployment on high-performance servers |
| PP-OCRv4_mobile_det | 77.79 | 10.6923 | 120.177 | 4.7 | PP-OCRv4's mobile text detection model, optimized for efficiency, suitable for deployment on edge devices |

**Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten texts, with 500 images for detection. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Text Recognition Module Models**:

<table>
    <tr>
        <th>Model</th>
        <th>Recognition Avg Accuracy (%)</th>
        <th>GPU Inference Time (ms)</th>
        <th>CPU Inference Time (ms)</th>
        <th>Model Size (M)</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>PP-OCRv4_mobile_rec</td>
        <td>78.20</td>
        <td>7.95018</td>
        <td>46.7868</td>
        <td>10.6 M</td>
        <td rowspan="2">PP-OCRv4 is the next version of Baidu PaddlePaddle's self-developed text recognition model PP-OCRv3. By introducing data augmentation schemes and GTC-NRTR guidance branches, it further improves text recognition accuracy without compromising inference speed. The model offers both server (server) and mobile (mobile) versions to meet industrial needs in different scenarios.</td>
    </tr>
    <tr>
        <td>PP-OCRv4_server_rec</td>
        <td>79.20</td>
        <td>7.19439</td>
        <td>140.179</td>
        <td>71.2 M</td>
    </tr>
</table>

**Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten texts, with 11,000 images for text recognition. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

<table >
    <tr>
        <th>Model</th>
        <th>Recognition Avg Accuracy (%)</th>
        <th>GPU Inference Time (ms)</th>
        <th>CPU Inference Time (ms)</th>
        <th>Model Size (M)</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>ch_SVTRv2_rec</td>
        <td>68.81</td>
        <td>8.36801</td>
        <td>165.706</td>
        <td>73.9 M</td>
        <td rowspan="1">
        SVTRv2 is a server-side text recognition model developed by the OpenOCR team at the Vision and Learning Lab (FVL) of Fudan University. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 6% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the A-list.
    </td>
    </tr>
</table>

**Note: The evaluation set for the above accuracy metrics is the [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task](https://aistudio.baidu.com/competition/detail/1131/0/introduction) A-list. GPU inference time is based on NVIDIA Tesla T4 with FP32 precision. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

<table >
    <tr>
        <th>Model</th>
        <th>Recognition Avg Accuracy (%)</th>
        <th>GPU Inference Time (ms)</th>
        <th>CPU Inference Time (ms)</th>
        <th>Model Size (M)</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>ch_RepSVTR_rec</td>
        <td>65.07</td>
        <td>10.5047</td>
        <td>51.5647</td>
        <td>22.1 M</td>
        <td rowspan="1">
        The RepSVTR text recognition model is a mobile-oriented text recognition model based on SVTRv2. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the B-list, while maintaining similar inference speed.
    </td>
    </tr>
</table>

**Note: The evaluation set for the above accuracy metrics is the [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task](https://aistudio.baidu.com/competition/detail/1131/0/introduction) B-list. GPU inference time is based on NVIDIA Tesla T4 with FP32 precision. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Formula Recognition Module Models**:

| Model Name | BLEU Score | Normed Edit Distance | ExpRate (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size |
|-|-|-|-|-|-|-|
| LaTeX_OCR_rec | 0.8821 | 0.0823 | 40.01 | - | - | 89.7 M |

**Note: The above accuracy metrics are measured on the [LaTeX-OCR Formula Recognition Test Set](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Seal Text Detection Module Models**:

| Model | Detection Hmean (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size (M) | Description |
|-------|---------------------|-------------------------|-------------------------|--------------|-------------|
| PP-OCRv4_server_seal_det | 98.21 | 84.341 | 2425.06 | 109 | PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers |
| PP-OCRv4_mobile_seal_det | 96.47 | 10.5878 | 131.813 | 4.6 | PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices |

**Note: The above accuracy metrics are evaluated on a self-built dataset containing 500 circular seal images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Text Image Rectification Module Models**:

| Model | MS-SSIM (%) | Model Size (M) | Description |
|-------|-------------|--------------|-------------|
| UVDoc | 54.40 | 30.3 M | High-precision text image rectification model |

**The accuracy metrics of the models are measured from the [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html).**

**Document Image Orientation Classification Module Models**:

| Model | Top-1 Acc (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size (M) | Description |
|-------|---------------|-------------------------|-------------------------|--------------|-------------|
| PP-LCNet_x1_0_doc_ori | 99.06 | 3.84845 | 9.23735 | 7 | A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, 270° |

**Note: The above accuracy metrics are evaluated on a self-built dataset covering various scenarios such as certificates and documents, containing 1000 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start

PaddleX provides pre-trained model pipelines that can be quickly experienced. You can experience the effect of the General Image Classification pipeline online, or locally using command line or Python.

Before using the General Layout Parsing pipeline locally, please ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md).

### 2.1 Experience via Command Line
One command is all you need to quickly experience the effect of the Layout Parsing pipeline. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png) and replace `--input` with your local path to make predictions.

```bash
paddlex --pipeline layout_parsing --input demo_paper.png --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the Layout Parsing pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default Layout Parsing pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to expand</summary>

```bash
paddlex --get_pipeline_config layout_parsing
```
After execution, the layout parsing pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config layout_parsing --save_path ./my_path
```

After obtaining the pipeline configuration file, you can replace `--pipeline` with the saved path of the configuration file to make it take effect. For example, if the configuration file is saved as `./layout_parsing.yaml`, simply execute:

```bash
paddlex --pipeline ./layout_parsing.yaml --input layout_parsing.jpg
```
Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If these parameters are still specified, the specified parameters will take precedence.

</details>

After running, the result will be:

<details>
   <summary> 👉Click to expand</summary>

```
{'input_path': PosixPath('/root/.paddlex/temp/tmp5jmloefs.png'), 'parsing_result': [{'input_path': PosixPath('/root/.paddlex/temp/tmpshsq8_w0.png'), 'layout_bbox': [51.46833, 74.22329, 542.4082, 232.77504], 'image': {'img': array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [213, 221, 238],
        [217, 223, 240],
        [233, 234, 241]],

       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8), 'image_text': ''}, 'layout': 'single'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpcd2q9uyu.png'), 'layout_bbox': [47.68295, 243.08054, 546.28253, 295.71045], 'figure_title': 'Overview of RT-DETR, We feed th', 'layout': 'single'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpr_iqa8b3.png'), 'layout_bbox': [58.416977, 304.1531, 275.9134, 400.07513], 'image': {'img': array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8), 'image_text': ''}, 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmphpblxl3p.png'), 'layout_bbox': [100.62961, 405.97458, 234.79774, 414.77414], 'figure_title': 'Figure 5. The fusion block in CCFF.', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmplgnczrsf.png'), 'layout_bbox': [47.81724, 421.9041, 288.01566, 550.538], 'text': 'D, Ds, not only significantly reduces latency (35% faster),\nRut\nnproves accuracy (0.4% AP higher), CCFF is opti\nased on the cross-scale fusion module, which\nnsisting of convolutional lavers intc\npath.\nThe role of the fusion block is t\n into a new feature, and its\nFigure 5. The f\nblock contains tw\n1 x1\nchannels, /V RepBlock\n. anc\n: two-path outputs are fused by element-wise add. We\ntormulate the calculation ot the hvbrid encoder as:', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpsq0ey9md.png'), 'layout_bbox': [94.60716, 558.703, 288.04193, 600.19434], 'formula': '\\begin{array}{l}{{\\Theta=K=\\mathrm{p.s.sp{\\pm}}\\mathrm{i.s.s.}(\\mathrm{l.s.}(\\mathrm{l.s.}(\\mathrm{l.s.}}),{\\qquad\\mathrm{{a.s.}}\\mathrm{s.}}}\\\\ {{\\tau_{\\mathrm{{s.s.s.s.s.}}(\\mathrm{l.s.},\\mathrm{l.s.},\\mathrm{s.s.}}\\mathrm{s.}\\mathrm{s.}}\\end{array}),}}\\\\ {{\\bar{\\mathrm{e-c.c.s.s.}(\\mathrm{s.},\\mathrm{s.s.},\\ s_{s}}\\mathrm{s.s.},\\tau),}}\\end{array}', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpv30qy0v4.png'), 'layout_bbox': [47.975555, 607.12024, 288.5776, 629.1252], 'text': 'tened feature to the same shape as Ss.\nwhere Re shape represents restoring the shape of the flat-', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp0jejzwwv.png'), 'layout_bbox': [48.383354, 637.581, 245.96404, 648.20496], 'paragraph_title': '4.3. Uncertainty-minimal Query Selection', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpushex416.png'), 'layout_bbox': [47.80134, 656.002, 288.50192, 713.24994], 'text': 'To reduce the difficulty of optimizing object queries in\nDETR, several subsequent works [42, 44, 45] propose query\nselection schemes, which have in common that they use the\nconfidence score to select the top K’ features from the en-\ncoder to initialize object queries (or just position queries).', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpki7e_6wc.png'), 'layout_bbox': [306.6371, 302.1026, 546.3772, 419.76724], 'text': 'The confidence score represents the likelihood that the fea\nture includes foreground objects. Nevertheless, the \nare required to simultaneously model the category\nojects, both of which determine the quality of the\npertor\ncore of the fes\nBased on the analysis, the current query\n considerable level of uncertainty in the\nresulting in sub-optimal initialization for\nand hindering the performance of the detector.', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmppbxrfehp.png'), 'layout_bbox': [306.0642, 422.7347, 546.9216, 539.45734], 'text': 'To address this problem, we propose the uncertainty mini\nmal query selection scheme, which explicitly const\noptim\n the epistemic uncertainty to model the\nfeatures, thereby providing \nhigh-quality\nr the decoder. Specifically,\n the discrepancy between i\nalization P\nand classificat\n.(2\ntunction for the gradie', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp1mgiyd21.png'), 'layout_bbox': [331.52808, 549.32635, 546.5229, 586.15546], 'formula': '\\begin{array}{c c c}{{}}&{{}}&{{\\begin{array}{c}{{i\\langle X\\rangle=({\\bar{Y}}({\\bar{X}})+{\\bar{Z}}({\\bar{X}})\\mid X\\in{\\bar{\\pi}}^{\\prime}}}&{{}}\\\\ {{}}&{{}}&{{}}\\end{array}}}&{{\\emptyset}}\\\\ {{}}&{{}}&{{C(\\bar{X},{\\bar{X}})=C..\\scriptstyle(\\bar{0},{\\bar{Y}})+{\\mathcal{L}}_{{\\mathrm{s}}}({\\bar{X}}),\\ 6)}}&{{}}\\end{array}', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp8t73dpym.png'), 'layout_bbox': [306.44016, 592.8762, 546.84314, 630.60126], 'text': 'where  and y denote the prediction and ground truth,\n= (c, b), c and b represent the category and bounding\nbox respectively, X represent the encoder feature.', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpftnxeyjm.png'), 'layout_bbox': [306.15652, 632.3142, 546.2463, 713.19073], 'text': 'Effectiveness analysis. To analyze the effectiveness of the\nuncertainty-minimal query selection, we visualize the clas-\nsificatior\nscores and IoU scores of the selected fe\nCOCO\na 12017, Figure 6. We draw the scatterplo\nt with\ndots\nrepresent the selected features from the model trained\nwith uncertainty-minimal query selection and vanilla query', 'layout': 'right'}]}
```
</details>

### 2.2 Python Script Integration
A few lines of code are all you need to quickly perform inference on your production line. Taking the general layout parsing pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing")

output = pipeline.predict("demo_paper.png")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the result as an image file
    res.save_to_xlsx("./output/")  # Save the result as an Excel file
    res.save_to_html("./output/")  # Save the result as an HTML file
```
The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX. | `str` | None |
| `device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `use_hpip` | Whether to enable high-performance inference, only available if the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the pipeline object to perform inference: The `predict` method takes `x` as a parameter, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| `str` | Supports passing the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png). |
| `str` | Supports passing a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing a dictionary type, where the key needs to correspond to the specific task, e.g., "img" for image classification tasks, and the value of the dictionary supports the above data types, e.g., `{"img": "/root/data1"}`. |
| `list` | Supports passing a list, where the list elements need to be of the above data types, e.g., `[numpy.ndarray, numpy.ndarray]`, `["/root/data/img1.jpg", "/root/data/img2.jpg"]`, `["/root/data1", "/root/data2"]`, `[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(4) Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving as files, with the supported file types depending on the specific pipeline, such as:

| Method | Description | Method Parameters |
|--------|-------------|-------------------|
| `save_to_img` | Saves the result as an image file. | `- save_path`: `str` type, the path to save the file. When it's a directory, the saved file name is consistent with the input file name. |
| `save_to_html` | Saves the result as an HTML file. | `- save_path`: `str` type, the path to save the file. When it's a directory, the saved file name is consistent with the input file name. |
| `save_to_xlsx` | Saves the result as an Excel file. | `- save_path`: `str` type, the path to save the file. When it's a directory, the saved file name is consistent with the input file name.

Within this tutorial on Artificial Intelligence and Computer Vision, we will explore the capabilities of saving and exporting results from various processes, including OCR (Optical Character Recognition), layout analysis, and table structure recognition. Specifically, the `save_to_img` function enables saving visualization results, `save_to_html` converts tables directly into HTML files, and `save_to_xlsx` exports tables as Excel files.

Upon obtaining the configuration file, you can customize various settings for the layout parsing pipeline by simply modifying the `pipeline` parameter within the `create_pipeline` method to point to your configuration file path.

For instance, if your configuration file is saved at `./my_path/layout_parsing.yaml`, you can execute the following code:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/layout_parsing.yaml")
output = pipeline.predict("layout_parsing.jpg")
for res in output:
    res.print()  # Prints the structured output of the layout parsing prediction
    res.save_to_img("./output/")  # Saves the img format results from each submodule of the pipeline
    res.save_to_xlsx("./output/")  # Saves the xlsx format results from the table recognition module
    res.save_to_html("./output/")  # Saves the html results from the table recognition module
```

## 3. Development Integration/Deployment

If the pipeline meets your requirements in terms of inference speed and accuracy, you can proceed with development integration or deployment.

To directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX offers three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In production environments, many applications require stringent performance metrics, especially response speed, to ensure efficient operation and smooth user experience. PaddleX provides a high-performance inference plugin that deeply optimizes model inference and pre/post-processing for significant end-to-end speedups. For detailed instructions on high-performance inference, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_deploy.md).

☁️ **Service Deployment**: Service deployment is a common form in production environments, where reasoning functions are encapsulated as services accessible via network requests. PaddleX enables cost-effective service deployment of pipelines. For detailed instructions on service deployment, refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/service_deploy.md).

Below are the API references and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- Both the response body and the request body for POST requests are JSON data (JSON objects).
- When the request is processed successfully, the response status code is `200`, and the response body attributes are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    | `errorCode` | `integer` | Error code. Fixed as `0`. |
    | `errorMsg` | `string` | Error description. Fixed as `"Success"`. |

    The response body may also have a `result` attribute, of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body attributes are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    | `errorCode` | `integer` | Error code. Same as the response status code. |
    | `errorMsg` | `string` | Error description. |

Operations provided by the service:

- **`infer`**

    Performs layout parsing.

    `POST /layout-parsing`

    - Request body attributes:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        | `file` | `string` | The URL of an image file or PDF file accessible by the service, or the Base64 encoded result of the content of the above-mentioned file types. For PDF files with more than 10 pages, only the content of the first 10 pages will be used. | Yes |
        | `fileType` | `integer` | File type. `0` indicates a PDF file, `1` indicates an image file. If this attribute is not present in the request body, the service will attempt to infer the file type automatically based on the URL. | No |
        | `useImgOrientationCls` | `boolean` | Whether to enable document image orientation classification. This function is enabled by default. | No |
        | `useImgUnwrapping` | `boolean` | Whether to enable text image rectification. This function is enabled by default. | No |
        | `useSealTextDet` | `boolean` | Whether to enable seal text detection. This function is enabled by default. | No |
        | `inferenceParams` | `object` | Inference parameters. | No |

        Attributes of `inferenceParams`:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        | `maxLongSide` | `integer` | During inference, if the length of the longer side of the input image for the text detection model is greater than `maxLongSide`, the image will be scaled so that the length of the longer side equals `maxLongSide`. | No |

    - When the request is processed successfully, the `result` of the response body has the following attributes:

        | Name | Type | Description |
        |------|------|-------------|
        | `layoutParsingResults` | `array` | Layout parsing results. The array length is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file. |

        Each element in `layoutParsingResults` is an `object` with the following attributes:

        | Name | Type | Description |
        |------|------|-------------|
        | `layoutElements` | `array` | Layout element information. |

        Each element in `layoutElements` is an `object` with the following attributes:

        | Name | Type | Description |
        |------|------|-------------|
        | `bbox` | `array` | Position of the layout element. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively. |
        | `label` | `string` | Label of the layout element. |
        | `text` | `string` | Text contained in the layout element. |
        | `layoutType` | `string` | Arrangement of the layout element. |
        | `image` | `string` | Image of the layout element, in JPEG format, encoded using Base64. |

</details>
</details>

<details>
<summary>Multi-language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/layout-parsing" # 服务URL

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64编码的文件内容或者文件URL
    "fileType": 1,
    "useImgOrientationCls": True,
    "useImgUnwrapping": True,
    "useSealTextDet": True,
}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
print("\nDetected layout elements:")
for res in result["layoutParsingResults"]:
    for ele in res["layoutElements"]:
        print("===============================")
        print("bbox:", ele["bbox"])
        print("label:", ele["label"])
        print("text:", repr(ele["text"]))
```

</details>
</details>
<br/>

📱 **Edge Deployment**: Edge deployment refers to placing computational and data processing capabilities directly on user devices, enabling them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy.md).

You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general layout parsing pipeline do not meet your requirements in terms of accuracy or speed for your specific scenario, you can try to further fine-tune the existing models using **your own domain-specific or application-specific data** to improve the recognition performance of the general layout parsing pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general layout parsing pipeline consists of 7 modules, unsatisfactory performance may stem from any of these modules.

You can analyze images with poor recognition results and follow the guidelines below for analysis and model fine-tuning:

* Incorrect table structure detection (e.g., wrong row/column recognition, incorrect cell positions) may indicate deficiencies in the table structure recognition module. You need to refer to the **Customization** section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md) and fine-tune the table structure recognition model using your private dataset.
* Misplaced layout elements (e.g., incorrect positioning of tables, seals) may suggest issues with the layout detection module. You should consult the **Customization** section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.md) and fine-tune the layout detection model with your private dataset.
* Frequent undetected texts (i.e., text missing detection) indicate potential weaknesses in the text detection model. Follow the **Customization** section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_detection.md) to fine-tune the text detection model using your private dataset.
* High text recognition errors (i.e., recognized text content does not match the actual text) suggest further improvements to the text recognition model. Refer to the **Customization** section in the [Text Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.md) to fine-tune the text recognition model.
* Frequent recognition errors in detected seal texts indicate the need for improvements to the seal text detection model. Consult the **Customization** section in the [Seal Text Detection Module Development Tutorials](../../../module_usage/tutorials/ocr_modules/) to fine-tune the seal text detection model.
* High recognition errors in detected formulas (i.e., recognized formula content does not match the actual formula) suggest further enhancements to the formula recognition model. Follow the [Customization](../../../module_usage/tutorials/ocr_modules/formula_recognition.md#Customization) section in the [Formula Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/formula_recognition.md) to fine-tune the formula recognition model.
* Frequent misclassifications of document or certificate orientations with text areas indicate the need for improvements to the document image orientation classification model. Refer to the **Customization** section in the [Document Image Orientation Classification Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md) to fine-tune the model.

### 4.2 Model Application
After fine-tuning your model with a private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the production line configuration file by replacing the local paths of the fine-tuned model weights to the corresponding positions in the configuration file:

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  # Can be modified to the local path of the fine-tuned model
  table_model: SLANet_plus  # Can be modified to the local path of the fine-tuned model
  text_det_model: PP-OCRv4_server_det  # Can be modified to the local path of the fine-tuned model
  text_rec_model: PP-OCRv4_server_rec  # Can be modified to the local path of the fine-tuned model
  formula_rec_model: LaTeX_OCR_rec  # Can be modified to the local path of the fine-tuned model
  seal_text_det_model: PP-OCRv4_server_seal_det   # Can be modified to the local path of the fine-tuned model
  doc_image_unwarp_model: UVDoc  # Can be modified to the local path of the fine-tuned model
  doc_image_ori_cls_model: PP-LCNet_x1_0_doc_ori  # Can be modified to the local path of the fine-tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
Subsequently, refer to the command line or Python script methods in the local experience to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference in the layout parsing pipeline, the Python command is:

```bash
paddlex --pipeline layout_parsing --input layout_parsing.jpg --device gpu:0
```
At this point, if you want to switch the hardware to Ascend NPU, simply modify `--device` to npu in the Python command:

```bash
paddlex --pipeline layout_parsing --input layout_parsing.jpg --device npu:0
```
If you want to use the general layout parsing pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.md).
