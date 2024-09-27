# PaddleX Table Structure Recognition Task Data Annotation Tutorial

## 1. Data Annotation
For annotating table data, use the [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel/blob/main/README_en.md) tool. Detailed steps can be referenced from: [【Video Demonstration】](https://www.bilibili.com/video/BV1wR4y1v7JE/?share_source=copy_web&vd_source=cf1f9d24648d49636e3d109c9f9a377d&t=1998)

Table annotation focuses on structured extraction of table data, converting tables in images into Excel format. Therefore, annotation requires the use of external software to open Excel simultaneously. In PPOCRLabel, complete the annotation of text information within the table (text and position), and in the Excel file, complete the annotation of table structure information. The recommended steps are:

1. Table Recognition: Open the table image, click the `Table Recognition` button in the upper right corner of the software. The software will call the table recognition model in PP-Structure to automatically label the table and simultaneously pop up Excel.
2. Modify Annotation Results: **Add annotation boxes with each cell as the unit** (i.e., all text within a cell is marked as one box). Right-click on the annotation box and select `Cell Re-recognition` to automatically recognize the text within the cell using the model.
3. **Adjust Cell Order**: Click `View - Show Box Number` in the software to display the annotation box numbers. Drag all results under the `Recognition Results` column on the right side of the software interface to arrange the annotation box numbers in order from left to right and top to bottom, annotating by row.
4. Annotate Table Structure: **In an external Excel software, mark cells with text with any identifier (e.g., `1`)**, ensuring that the cell merging in Excel matches the original image (i.e., the text in Excel cells does not need to be identical to the text in the image).
5. Export JSON Format: Close all Excel files corresponding to the table images, click `File - Export Table Annotations`, and generate the gt.txt annotation file.

## 2. Data Format
The dataset structure and annotation format defined by PaddleX for table recognition tasks are as follows:

```ruby
dataset_dir    # Root directory of the dataset, the directory name can be changed
├── images     # Directory for saving images, the directory name can be changed, but note the correspondence with the content of train.txt and val.txt
├── train.txt  # Training set annotation file, the file name cannot be changed. Example content: {"filename": "images/border.jpg", "html": {"structure": {"tokens": ["<tr>", "<td", " colspan=\"3\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>"]}, "cells": [{"tokens": ["、", "自", "我"], "bbox": [[[5, 2], [231, 2], [231, 35], [5, 35]]]}, {"tokens": ["9"], "bbox": [[[168, 68], [231, 68], [231, 98], [168, 98]]]}]}, "gt": "<html><body><table><tr><td colspan=\"3\">、自我</td></tr><tr><td>Aghas</td><td>失吴</td><td>月，</td></tr><tr><td>lonwyCau</td><td></td><td>9</td></tr></table></body></html>"}
└── val.txt    # Validation set annotation file, the file name cannot be changed. Example content: {"filename": "images/no_border.jpg", "html": {"structure": {"tokens": ["<tr>", "<td", " colspan=\"2\"", ">", "</td>", "<td", " rowspan=\"2\"", ">", "</td>", "<td", " rowspan=\"2\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>"]}, "cells": [{"tokens": ["a", "d", "e", "s"], "bbox": [[[0, 4], [284, 4], [284, 34], [0, 34]]]}, {"tokens": ["$", "7", "5", "1", "8", ".", "8", "3"], "bbox": [[[442, 67], [616, 67], [616, 100], [442, 100]]]}]}, "gt": "<html><body><table><tr><td colspan=\"2\">ades</td><td rowspan=\"2\">研究中心主任滕建</td><td rowspan=\"2\">品、家居用品位居商</td></tr><tr><td>naut</td><td>则是创办思</td></tr><tr><td>各方意见建议，确保</td><td>9.66</td><td>道开业，负责</td><td>$7518.83</td></tr></table></body></html>"}
```
Annotation files are labeled in the `PubTabNet` dataset format, with each line containing a `python` dictionary.

Please prepare your data according to the above specifications. Additionally, you can refer to: [Example Dataset](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar)