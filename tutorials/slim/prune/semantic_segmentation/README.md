# Semantic segmentation model pruning training

## Step 1: Perform the normal training semantic segmentation model.

```
python unet_train.py
```

In this step, the training model is saved in the `output/unet` directory.

## Step 2: Analyze the model parameter information.

```
python param_analysis.py
```
After the parameters have been analyzed, you can get the `unet.sensi.data` file. It saves the sensitive information for each parameter. `

> You can continue to load the model and sensitive file for visualization by running the following command:
> ```
> python slim_visualize.py
> ```
> The visualization results are as follows: 
The vertical axis is `eval_metric_loss` (the parameter to be configured in Step 3) and the horizontal axis is the scale at which the model is pruned. See it in the following diagram.
- When `eval_metric_loss` is set to 0.05, the model is pruned by 64.1% (35.9% of the model remains).
- When `eval_metric_loss` is set to 0.1, the model is pruned by 70.9% (29.1% of the model remains).

![](./sensitivities.png)

## Step 3: Perform the model pruning training.

```
python unet_prune_train.py
```
The code in this step is almost identical to the code in Step 1, and the only difference is: in the last train function, the four parameters such as ` pretrain_weights`, `save_dir`, `sensitivities_file`, and `eval_metric_loss` in `unet_prune_train.py`` are modified.

- pretrain_weights: In the pruning training, set to the previously trained model.
- save_dir: It indicates the model storage location in the model training process.
- sensitivities_file: It is the parameter sensitive information file obtained from the analysis in Step 2.
- eval_metric_loss: It is the visualized relevant parameter in Step 2. You can use the parameter to change the pruning scale of the final model accordingly.

## Pruning effect

For data in this example, the pruning effects are compared as follows: The prediction is performed by using the **CPU, with disabling MKLDNN**. The prediction time does not include pre-processing of data or post-processing of the results. 
It can be seen that after the model is pruned by 64%, the model precision remains essentially unchanged and the prediction time for a single image is reduced by almost 50%.

> The UNet model is used here for comparison only. In fact, on low performance devices, it is recommended to use lightweight segmentation models such as deeplab-mobilenet or fastscnn.

| Model | Parameter file size | Prediction speed | mIOU |
| :--- | :----------  | :------- | :--- |
| UNet |    52M       |   9.85s  | 0.915 |
| UNet (pruned by 64%) | 19M | 4.80s | 0.911 |
