# Image classification model quantification

A quantification example of the MobileNetV2 model is provided in this directory by executing the following command:

## Step 1: Obtain the quantification model.
```
python mobilenetv2_quant.py
```
Execute the code to automatically download the model and data set

## Step 2: Export as the PaddleLite model.

```
python paddlelite_export.py
```
Before executing this script, you need to install paddlelite. In the Python environment, run `pip install paddlelite`.
