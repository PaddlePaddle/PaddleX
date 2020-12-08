# Model interpretability

Currently, PaddleX supports visual interpretation of image classification results and supports LIME and NormLIME interpretability algorithms.

## paddlex.interpret.lime
> **Visualization of LIME interpretability results**
```
paddlex.interpret.lime(img_file,
                       model,
                       num_samples=3000,
                       batch_size=50,
                       save_dir='./')
```
The LIME algorithm is used to visualize the interpretability of model prediction results. LIME represents model-independent local interpretability and can interpret any model. The idea of LIME is as follows: By taking an input sample as a center and randomly taking a sample in the space near it, each sample obtains a new output through the original model, so a series of inputs and the corresponding outputs are obtained. LIME fits this mapping relation using a simple and interpretable model (such as a linear regression model) to get the weight of each input dimension to interpret the model.

**Note: **Currently, the visualization of interpretability results supports classification models only. 

### Parameters
> * **img_file** (str): Prediction image path.
> * **model** (paddlex.cv.models): Model in paddlex.
> * **num_samples** (int): Number of samples that LIME uses for linear learning models. It is 3000 by default.
> * **batch_size** (int): Prediction data batch size. It is 50 by default.
> * **save_dir** (str): Storage path of visualized interpretability results (saved as a png file) and intermediate files.


### Visualization effects

![](./docs/gui/images/LIME.png)

### Usage example

> For the visualization process of prediction interpretability results, refer to [codes] (https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/interpret/lime.py).


## paddlex.interpret.normlime
> **Visualization of NormLIME interpretability results**
```
paddlex.interpret.normlime(img_file,
                           model,
                           dataset=None,
                           num_samples=3000,
                           batch_size=50,
                           save_dir='./',
                           normlime_weights_file=None)
```
The NormLIME algorithm is used to visualize the interpretability of model prediction results.
NormLIME uses a certain number of samples to make a global interpretation. A simplified method is used here because the NormLIME calculations are large: Use a certain number of test samples (Currently, all test samples are used by default) to perform feature extractions on each sample and map them to the same feature space. By using this feature as an input and the model output as an output, use linear regression to fit it to obtain a global input and output relation. When a test sample is interpreted, use NormLIME global interpretation to filter LIME results so that the final visualized results are more stable.

**Note: **Currently, the visualization of interpretability results supports classification models only.

### Parameters
> * **img_file** (str): Prediction image path.
> * **model** (paddlex.cv.models): Model in paddlex.
> * **dataset** (paddlex.datasets): Dataset reader. It is none by default.
> * **num_samples** (int): Number of samples that LIME uses for linear learning models. It is 3000 by default.
> * **batch_size** (int): Prediction data batch size. It is 50 by default.
> * **save_dir** (str): Storage path of visualized interpretability results (saved as a png file) and intermediate files.
> * **normlime_weights_file** (str): NormLIME initialization filename. If it does not exist, it is calculated once and saved in this path. If it exists, it is directly loaded.


**Note: **`dataset` reads a dataset. This dataset shall not be too large, otherwise the calculation time is long. However, all data categories shall be contained. Currently, the visualization of NormLIME interpretability results supports classification models only.
### Usage example
> For the visualization process of prediction interpretability results, refer to [codes] (https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/interpret/normlime.py).
