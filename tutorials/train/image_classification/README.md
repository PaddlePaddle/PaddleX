# Image classification training example

The image classification example codes are in this directory. Users can perform the training directly after installing PaddlePaddle and PaddleX.

- [PaddlePaddle installation](https://www.paddlepaddle.org.cn/install/quick)
- [PaddleX installation](https://paddlex.readthedocs.io/zh_CN/develop/install.html)

## Model training
As shown below, you can download the codes and run them directly. The training data is downloaded automatically through the codes.
```
python mobilenetv3_small_ssld.py
```

## VisualDL visual training index
In the model training process, if you set `use_vdl` to True in the `train` function, the training log is automatically marked into the `vdl_log` directory under `save_dir` (the path specified by the user) in the format of VisualDL, to view the visualized index.
```
visualdl --logdir output/mobilenetv3_small_ssld/vdl_log --port 8001
```

After the service starts, use your browser to visit the website at https://0.0.0.0:8001 or https://localhost:8001.
