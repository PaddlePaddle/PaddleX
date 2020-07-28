# 模型可解释性

目前PaddleX支持对于图像分类的结果以可视化的方式进行解释，支持LIME和NormLIME两种可解释性算法。

## paddlex.interpret.lime
> **LIME可解释性结果可视化**  
```
paddlex.interpret.lime(img_file,
                       model,
                       num_samples=3000,
                       batch_size=50,
                       save_dir='./')
```
使用LIME算法将模型预测结果的可解释性可视化。  
LIME表示与模型无关的局部可解释性，可以解释任何模型。LIME的思想是以输入样本为中心，在其附近的空间中进行随机采样，每个采样通过原模型得到新的输出，这样得到一系列的输入和对应的输出，LIME用一个简单的、可解释的模型（比如线性回归模型）来拟合这个映射关系，得到每个输入维度的权重，以此来解释模型。  

**注意：** 可解释性结果可视化目前只支持分类模型。

### 参数
>* **img_file** (str): 预测图像路径。
>* **model** (paddlex.cv.models): paddlex中的模型。
>* **num_samples** (int): LIME用于学习线性模型的采样数，默认为3000。
>* **batch_size** (int): 预测数据batch大小，默认为50。
>* **save_dir** (str): 可解释性可视化结果（保存为png格式文件）和中间文件存储路径。


### 使用示例
> 对预测可解释性结果可视化的过程可参见[代码](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/interpret/lime.py)。


## paddlex.interpret.normlime
> **NormLIME可解释性结果可视化**  
```
paddlex.interpret.normlime(img_file,
                           model,
                           dataset=None,
                           num_samples=3000,
                           batch_size=50,
                           save_dir='./',
                           normlime_weights_file=None)
```
使用NormLIME算法将模型预测结果的可解释性可视化。
NormLIME是利用一定数量的样本来出一个全局的解释。由于NormLIME计算量较大，此处采用一种简化的方式：使用一定数量的测试样本（目前默认使用所有测试样本），对每个样本进行特征提取，映射到同一个特征空间；然后以此特征做为输入，以模型输出做为输出，使用线性回归对其进行拟合，得到一个全局的输入和输出的关系。之后，对一测试样本进行解释时，使用NormLIME全局的解释，来对LIME的结果进行滤波，使最终的可视化结果更加稳定。

**注意：** 可解释性结果可视化目前只支持分类模型。

### 参数
>* **img_file** (str): 预测图像路径。
>* **model** (paddlex.cv.models): paddlex中的模型。
>* **dataset** (paddlex.datasets): 数据集读取器，默认为None。
>* **num_samples** (int): LIME用于学习线性模型的采样数，默认为3000。
>* **batch_size** (int): 预测数据batch大小，默认为50。
>* **save_dir** (str): 可解释性可视化结果（保存为png格式文件）和中间文件存储路径。
>* **normlime_weights_file** (str): NormLIME初始化文件名，若不存在，则计算一次，保存于该路径；若存在，则直接载入。

**注意：** `dataset`读取的是一个数据集，该数据集不宜过大，否则计算时间会较长，但应包含所有类别的数据。NormLIME可解释性结果可视化目前只支持分类模型。
### 使用示例
> 对预测可解释性结果可视化的过程可参见[代码](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/interpret/normlime.py)。
