# Model compression

## paddlex.slim.cal_params_sensitivities
> **Calculate parameter sensitivity**
```
paddlex.slim. cal_params_sensitivities(model, save_file, eval_dataset, batch_size=8)
```
Calculate sensitivity of pruned parameters in the model on the validation set and save sensitivity information in the `save_file` file
1. Obtain the name of the pruned convolutions Kernel in the model.
2. Calculate sensitivity of pruned convolutions Kernel at different pruning rates.

[Note] Convolution sensitivity is a model precision loss after the model is pruned according to a pruning rate. Select an appropriate sensitivity and thus determine the list of parameters to be pruned for the final model and the pruning rate corresponding to each pruned parameter.

[View an example](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/compress/classification/cal_sensitivities_file.py#L33)

**Parameters**

* **model** (paddlex.cls.models/paddlex.det.models/paddlex.seg.models): Model loaded by paddlex.
* **save_file** (str): Storage path of the calculated parameter sensitivity file.
* **eval_dataset** (paddlex.datasets): Reader for evaluated datasets.
* **batch_size** (int): batch_size size during evaluation.


## paddlex.slim.export_quant_model
> **Export a quantitative model**
```
paddlex.slim. export_quant_model(model, test_dataset, batch_size=2, batch_num=10, save_dir='. /quant_model', cache_dir='. /temp')
```
Export a quantitative model. This API implements the Post Quantization quantization method. An incoming test dataset is required. In addition, `batch_size` and `batch_num` need to be set. The calculation results of sample data of which the quantity is `batch_size` * ` `batch_num` are used as statistic information to complete the quantization of the model during quantization.

**Parameters**

* **model** (paddlex.cls.models/paddlex.det.models/paddlex.seg.models): Model loaded by paddlex.
* **test_dataset**(paddlex.dataset): Test dataset.
* **batch_size**(int): Batch data size during the forward calculation.
* **batch_num**(int): Batch data quantity during the forward calculation.
* **save_dir**(str): Directory where a quantized model is saved.
* **cache_dir**(str): Temporary storage directory of statistic data during quantization.


**Usage example**

Click to download the [model](https://bj.bcebos.com/paddlex/models/vegetables_mobilenetv2.tar.gz) and [dataset](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz) in the following example
```
import paddlex as pdx
model = pdx.load_model('vegetables_mobilenet')
test_dataset = pdx.datasets.ImageNet(
                    data_dir='vegetables_cls',
                    file_list='vegetables_cls/train_list.txt',
                    label_list='vegetables_cls/labels.txt',
                    transforms=model.eval_transforms)
pdx.slim.export_quant_model(model, test_dataset, save_dir='./quant_mobilenet')
```