# PaddleX图像识别任务模块数据标注教程

该部分将介绍如何使用[Labelme](https://github.com/wkentaro/labelme)标注工具完成图像识别相关单模型的数据标注。 
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程。

## 1. Labelme 标注
### 1.1 Labelme 标注工具介绍
`Labelme` 是一个 `python` 语言编写，带有图形界面的图像标注软件。可用于图像分类、目标检测、图像分割等任务，在图像识别的标注任务中，标签存储为 `JSON` 文件。

### 1.2 Labelme 安装
为避免环境冲突，建议在 `conda` 环境下安装。

```ruby
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```
### 1.3 Labelme 标注过程
#### 1.3.1 准备待标注数据
* 创建数据集根目录，如 `pets`。
* 在 `pets` 中创建 `images` 目录（必须为`images`目录），并将待标注图片存储在 `images` 目录下，如下图所示：

![alt text](/tmp/images/data_prepare/image_classification/01.png)

* 在 `pets` 文件夹中创建待标注数据集的类别标签文件 `flags.txt`，并在 `flags.txt` 中按行写入待标注数据集的类别。以猫狗分类数据集的 `flags.txt` 为例，如下图所示：

![alt text](/tmp/images/data_prepare/image_classification/02.png)
#### 1.3.2 启动 Labelme
终端进入到待标注数据集根目录，并启动 `labelme` 标注工具。

```ruby
cd path/to/pets
labelme images --nodata --autosave --output annotations --flags flags.txt
```
* `flags` 为图像创建分类标签，传入标签路径。
* `nodata` 停止将图像数据存储到 JSON 文件。
* `autosave` 自动存储。
* `ouput` 标签文件存储路径。
#### 1.3.3 开始图片标注
* 启动 `labelme` 后如图所示：

![alt text](/tmp/images/data_prepare/image_classification/03.png)
* 在 `Flags` 界面选择类别。

![alt text](/tmp/images/data_prepare/image_classification/04.png)

* 标注好后点击存储。（若在启动 `labelme` 时未指定 `output` 字段，会在第一次存储时提示选择存储路径，若指定 `autosave` 字段使用自动保存，则无需点击存储按钮）。

![alt text](/tmp/images/data_prepare/image_classification/05.png)
* 然后点击 `Next Image` 进行下一张图片的标注。

![alt text](/tmp/images/data_prepare/image_classification/06.png)

* 完成全部图片的标注后，使用[convert_to_imagenet.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/convert_to_imagenet.py)脚本将标注好的数据集转换为 `ImageNet-1k` 数据集格式，生成 `train.txt`，`val.txt` 和`label.txt`。

```python
python convert_to_imagenet.py --dataset_path /path/to/dataset
```
`dataset_path`为标注的 `labelme` 格式分类数据集。

* 经过整理得到的最终目录结构如下：

![alt text](/tmp/images/data_prepare/image_classification/07.png)

#### 1.3.4 数据格式转换
在获得 `LabelMe` 格式数据后，需要将数据格式转换为`ShiTuRecDataset`格式。下面给出了按照上述教程使用`LableMel`标注完成的数据并进行数据格式转换的代码示例。

```ruby
# 下载并解压 LabelMe 格式示例数据集
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/image_classification_labelme_examples.tar -P ./dataset
tar -xf ./dataset/image_classification_labelme_examples.tar -C ./dataset/
#将 LabelMe 示例数据集进行转换
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
```
## 3. 数据格式
PaddleX 针对图像分类任务定义的数据集，名称是 **ShiTuRecDataset**，组织结构和标注格式如下：

```ruby
dataset_dir    # 数据集根目录，目录名称可以改变
├── images     # 图像的保存目录，目录名称可以改变，但要注意与train.txt、query.txt、 gallery.txt 的内容对应
├── gallery.txt   # 验证集标注文件，文件名称不可改变。每行给出待检索图像路径和图像特征标签，使用空格分隔，内容举例：images/WOMEN/Blouses_Shirts/id_00000001/02_2_side.jpg 3997
└── query.txt     # 验证集标注文件，文件名称不可改变。每行给出数据库图像路径和图像特征标签，使用空格分隔，内容举例：images/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg 3997
```
标注文件采用图像特征格式。请大家参考上述规范准备数据，此外可以参考[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Inshop_examples.tar)。