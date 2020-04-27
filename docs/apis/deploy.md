# paddlex.deploy

使用AnalysisPredictor进行预测部署。

## create_predictor

```
paddlex.deploy.create_predictor(model_dir, use_gpu=False)
```

#### Args

* **model_dir**: 训练过程中保存的模型路径
* **use_gpu**: 是否使用GPU进行预测

#### Returns

* **Predictor**: paddlex.deploy.predictor.Predictor

### 示例

```
import paddlex
# 下
Predictor = paddlex.deploy.create_predictor(model_dir, use_gpu=True)
```

## ClassifyPredictor
继承至paddlex.deploy.predictor.Predictor，当model_dir/model.yml里面

```
paddlex.deploy.create_predictor(model_dir, use_gpu=False)
```

#### Args

* **model_dir**: 训练过程中保存的模型路径
* **use_gpu**: 是否使用GPU进行预测

#### Returns

* **Predictor**: paddlex.deploy.predictor.Predictor

### 示例

```
import paddlex
# 下
Predictor = paddlex.deploy.create_predictor(model_dir, use_gpu=True)
```
