# 多模型组合

PaddleX 提供了十分丰富的模型，以及针对不同任务的模型产线，同时PaddleX也支持用户多模型组合使用，以解决复杂、特定的任务。

## 一、任务分析

下面通过组合使用PaddleX提供的版面分析模型、表格识别模型和OCR产线，解决表格识别任务，具体来说，该任务分为以下几个步骤：

1. 使用版面分析模型，检测出文档图片中的表格区域位置；
2. 使用OpenCV等操作，裁剪出文档图片中表格区域图片；
3. 使用表格识别模型，对表格图片进行识别，得到表格结构的html表示；
4. 使用OCR产线，对表格图片进行识别，得到表格区域的文字；

## 二、示例代码

根据任务分析，即可使用PaddleX进行开发，完整示例代码如下：

```python
import cv2
from paddlex import create_model, create_pipeline


class TableRec:

    def __init__(self):
        self.layout_model = create_model("PicoDet_layout_1x")
        self.table_model = create_model("SLANet_plus")
        self.ocr_pipeline = create_pipeline("OCR")

    def crop_table(self, layout_res):
        img_path = layout_res["input_path"]
        img = cv2.imread(img_path)

        table_img_list = []
        for box in layout_res["boxes"]:
            if box["label"] != "Table":
                continue
            xmin, ymin, xmax, ymax = [int(i) for i in box["coordinate"]]
            table_img = img[ymin:ymax, xmin:xmax]
            table_img_list.append({"input": table_img})
        return table_img_list

    def predict(self, data):
        for layout_res in self.layout_model(data):
            final_res = {}
            table_img_list = self.crop_table(layout_res)
            table_res = list(self.table_model(table_img_list))
            ocr_res = list(self.ocr_pipeline(table_img_list))
            final_res["structure"] = table_res["structure"]
            final_res["ocr_box"] = ocr_res["dt_polys"]
            final_res["rec_text"] = ocr_res["rec_text"]
            yield final_res


if __name__ == "__main__":
    solution = TableRec()
    output = solution.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg")
    for res in output:
        print(res)
```

## 三、代码详解

接下来对以上代码进行详解：

1. 实例化模型和产线

```python
self.layout_model = create_model("PicoDet_layout_1x")   # 实例化版面分析模型
self.table_model = create_model("SLANet_plus")          # 实例化表格识别模型
self.ocr_pipeline = create_pipeline("OCR")              # 实例化OCR模型产线
```

PaddleX提供了`create_model(model)`函数用于实例化模型，只需通过参数`model`指定模型名称即可使用PaddleX提供的官方预训练模型，或是通过参数`model`指定本地模型路径，即可使用训练好的本地模型。关于PaddleX支持的全部官方预训练模型请参考[模型列表](../../support_list/models_list.md)，关于`create_model`方法的更详细使用方式，请参考文档[模型使用 Python API](./model_python_API.md)。

PaddleX提供了`create_pipeline(pipeline)`函数用于实例化产线，只需通过参数`pipeline`指定产线名称即可使用PaddleX提供的模型产线，或是通过参数`pipeline`指定本地产线配置文件（`*.yaml`），即可使用自定义的模型产线。关于PaddleX内置的模型产线可以查看[产线列表](../../support_list/pipelines_list.md)，关于`create_pipeline`方法的更详细使用方式，请参考文档[产线使用 Python API](../../pipeline_usage/instructions/pipeline_python_API.md)。

2. 调用版面分析模型

```python
for layout_res in self.layout_model(data):
    pass
```

PaddleX推理预测功能中的模型类（`Predictor`）和产线类（`Pipeline`）均实现了`__call__(input)`和`predict(input)`方法，支持通过参数`input`传入待预测数据，同时上述两种方法均基于`yield`实现，因此需要作为`generator`调用。关于模型预测和产线预测的详细说明请参考[模型使用 Python API](./model_python_API.md)和[产线使用 Python API](../../pipeline_usage/instructions/pipeline_python_API.md)。

对于版面分析模型`self.layout_model`，首先传入待预测数据`data`，并通过`for-in`的方式得到每张图片的版面分析预测结果`layout_res`。

3. 处理版面分析预测结果

```python
table_img_list = self.crop_table(layout_res)


def crop_table(self, layout_res):
    img_path = layout_res["input_path"]
    img = cv2.imread(img_path)

    table_img_list = []
    for box in layout_res["boxes"]:
        if box["label"] != "Table":
            continue
        xmin, ymin, xmax, ymax = [int(i) for i in box["coordinate"]]
        table_img = img[ymin:ymax, xmin:xmax]
        table_img_list.append({"input": table_img})
    return table_img_list
```

在得到版面分析预测结果`layout_res`后，需要对其进行处理，并构造符合后续模型预测输入格式的数据。

首先读取原始图像`img`，然后依据预测结果中每个目标（`box`）的类别信息（`box["label"]`）和位置坐标信息（`box["coordinate"]`），从原始图像裁剪（`img[ymin:ymax, xmin:xmax]`）得到表格子图（`table_img`），表格子图就是后续表格识别模型和OCR模型的待预测数据，因此需要将待预测数据按要求进行整理，具体来说需要整理为`{"input": table_img}`形式的字典，其中字典的key必须为`input`，因为PaddleX中模型或产线的预测输入函数的参数即为`input`，而对应的value即为待预测数据，如果需要预测一批数据，则应为上述字典组成的list（`table_img_list`）。

4. 调用表格识别模型与OCR模型

```python
table_res = list(self.table_model(table_img_list))
ocr_res = list(self.ocr_pipeline(table_img_list))
final_res["structure"] = table_res["structure"]
final_res["ocr_box"] = ocr_res["dt_polys"]
final_res["rec_text"] = ocr_res["rec_text"]
```

得到待预测的表格图像（`table_img_list`）后，即可进行表格结构识别和OCR识别，得到所需预测结果。
