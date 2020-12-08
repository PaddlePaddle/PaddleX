# Image classification model pruning training

## Step 1: Perform the normal training image classification model.

```
python mobilenetv2_train.py
```

In this step, the training model is saved in the `output/mobilenetv2` directory.

## Step 2: Analyze the model parameter information.

```
python param_analysis.py
```
After the parameters are analyzed, you get the `mobilenetv2.sensi.data` file. It saves the sensitive information for each parameter. 

> You can continue to load the model and sensitive file for visualization by running the following command:
> ```
> python slim_visualize.py
> ```
> The visualization results are as follows: 
The vertical axis is `eval_metric_loss` (the parameter to be configured in Step 3) and the horizontal axis is the scale at which the model is pruned. See it in the following diagram.
- When `eval_metric_loss` is set to 0.05, the model is pruned by 68.4% (31.6% of the model remains)
- When `eval_metric_loss` is set to 0.1, the model is pruned by 78.5% (21.5% of the model remains)

![](./sensitivities.png)

## Step 3: Perform the model pruning training.

```
python mobilenetv2_prune_train.py
```
The code in this step is almost identical to that in Step 1. The only difference is: in the last train function, the four parameters such as ` pretrain_weights`, `save_dir`, `sensitivities_file`, and `eval_metric_loss` in `mobilenetv2_prune_train.py` are modified.

- pretrain_weights: In the pruning training, set to the previously trained model.
- save_dir: It indicates the model storage location in the model training process.
- sensitivities_file: It is the parameter sensitive information file obtained from the analysis in Step 2.
- eval_metric_loss: It is the visualized relevant parameter in Step 2. You can use the parameter to change the pruning scale of the final model accordingly.


## Pruning effect

For data in this example, the pruning effects are compared as follows: The prediction is performed by using the **CPU, with disabling MKLDNN**. The prediction time does not include pre-processing of data or post-processing of the results.
It can be seen that after the model is pruned by 64%, the model precision increases and the prediction time for a single image is reduced by 37%.


| Model | Parameter file size | Prediction speed | Accuracy rate |
| :--- | :----------  | :------- | :--- |
| MobileNetV2 |    8.7M       |   0.057s  | 0.92 |
| MobileNetV2 (pruned by 68%) | 2.8M | 0.036s | 0.99 |
