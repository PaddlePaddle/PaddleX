# Object detection model pruning training

## Step 1: Perform the normal training object detection model.

```
python yolov3_train.py
```

In this step, the training model is saved in the `output/yolov3_mobilenetv1` directory.

## Step 2: Analyze the model parameter information.

```
python param_analysis.py
```
After the parameters are analyzed, you can get the `yolov3.sensi.data` file. It saves the sensitive information for each parameter. `

> You can continue to load the model and sensitive file for visualization by running the following command:
> ```
> python slim_visualize.py
> ```
> The visualization results are as follows: 
The vertical axis is `eval_metric_loss` (the parameter to be configured in Step 3) and the horizontal axis is the scale at which the model is pruned. See it in the following diagram.
- When `eval_metric_loss` is set to 0.05, the model is pruned by 63.1% (36.9% of the model remains)
- When `eval_metric_loss` is set to 0.1, the model is pruned by 68.6% (31.4% of the model remains)

![](./sensitivities.png)

## Step 3: Perform the model pruning training.

```
python yolov3_prune_train.py
```
The code for this step is almost identical to that in Step 1. The only difference is: in the last train function, the four parameters such as ` pretrain_weights`, `save_dir`, `sensitivities_file`, and `eval_metric_loss` in `yolov3_prune_train.py` are modified.

- pretrain_weights: In the pruning training, set to the previously trained model.
- save_dir: It indicates the model storage location in the model training process.
- sensitivities_file: It is the parameter sensitive information file obtained from the analysis in Step 2.
- eval_metric_loss: It is the visualized relevant parameter in Step 2. You can use the parameter to change the pruning scale of the final model accordingly.

## Pruning effect

For data in this example, the pruning effects are compared as follows: The prediction is performed by using the **CPU, with disabling MKLDNN**. The prediction time does not include pre-processing of data or post-processing of the results.
It can be seen that after the model is pruned by 63%, the model precision increases and the prediction time for a single image is reduced by 30%.


| Model | Parameter file size | Prediction speed | MAP |
| :--- | :----------  | :------- | :--- |
| YOLOv3-MobileNetV1 |    93M       |   1.045s  | 0.635 |
| YOLOv3-MobileNetV1 (pruned by 63%) | 35M | 0.735s | 0.735 |
