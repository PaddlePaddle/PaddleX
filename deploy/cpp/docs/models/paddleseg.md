# PaddleSeg模型部署

当前支持PaddleSeg release/2.2分支训练的模型进行导出及部署。本文档以[Deeplabv3P](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.2/configs/deeplabv3p)模型为例，讲述从release-2.2版本导出模型并进行cpp部署整个流程。 PaddleSeg相关详细文档查看[官网文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.2/README_CN.md)

## 步骤一 部署模型导出

### 1.获取PaddleSeg源码

```sh
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
```

### 2. 导出基于Cityscape数据的预训练模型

在进入`PaddleSeg`目录后，执行如下命令导出预训练模型

```sh
# 下载预训练权重
wget https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_769x769_80k/model.pdparams
# 导出部署模型
python export.py --config configs/deeplabv3p/deeplabv3p_resnet101_os8_cityscapes_769x769_80k.yml \
                 --model_path ./model.pdparams \
                 --save_dir output
```

导出的部署模型会保存在`output`目录，其结构如下

```
output
  ├── deploy.yaml            # 模型配置文件信息
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```

## 步骤二 编译

参考编译文档

- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南(生成exe)](../compile/paddle/windows.md)
- [Windows系统上编译指南(生成dll供C#调用)](../../../../examples/C%23_deploy/)

## 步骤三 模型预测

编译后即可获取可执行的二进制demo程序`model_infer`和`multi_gpu_model_infer`，分别用于在单卡/多卡上加载模型进行预测，对于分类模型，调用如下命令即可进行预测

```
./build/demo/model_infer --model_filename=output/model.pdmodel \
                         --params_filename=output/model.pdiparams \
                         --cfg_file=output/deploy.yaml \
                         --image=test.jpg \
                         --model_type=seg
```

输出结果如下(由于分割结果的score_map和label_map不便于直接输出，因此在demo程序中仅输出这两个mask的均值和方差)

```
ScoreMask(mean: 12.4814 std:    10.4955)    LabelMask(mean: 1.98847 std:    10.3141)
```

**注意：**release/2.1之后，PaddleSeg导出的模型默认只有label_map, score_map的值都被填充为1.0

关于demo程序的详细使用方法可分别参考以下文档

- [单卡加载模型预测示例](../demo/model_infer.md)
- [多卡加载模型预测示例](../demo/multi_gpu_model_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](../../demo/tensorrt_infer.md)
