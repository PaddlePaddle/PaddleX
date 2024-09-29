# Image Unwarping Module Development Tutorial

## I. Overview
The primary purpose of image unwarping is to perform geometric transformations on images in order to correct issues such as document distortion, tilt, perspective deformation, etc., enabling more accurate recognition by subsequent text recognition modules.

## II. Supported Model List

|Model Name|MS-SSIM （%）|Model Size (M)| information|
|-|-|-|-|
|UVDoc |54.4|31 M|High-precision Document Image Unwarping Model|


**The accuracy metrics of the above models are measured on the [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset.**

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md)


Just a few lines of code can complete the inference of the image unwarping module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the image unwarping module into your project.

```bash
from paddlex import create_model
model = create_model("UVDoc")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
The current module temporarily does not support fine-tuning training and only supports inference integration. Fine-tuning training for this module is planned to be supported in the future.
