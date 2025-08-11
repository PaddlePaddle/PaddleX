---
comments: true
---

# FAQ

## Q: What is PaddleX?

A: PaddleX is a low-code development tool featuring selected models and pipelines launched by the PaddlePaddle team. It aims to provide developers with a more convenient and efficient development environment. This tool supports model training and inference on multiple mainstream domestic and international hardware, and is compatible with various system configurations, thereby meeting users' needs in different application scenarios. PaddleX has a wide range of applications, covering industries such as industry, energy, finance, transportation, and education, providing professional support and solutions for these industries. By using PaddleX, developers can more easily apply deep learning technology to actual industrial practices, thereby realizing technology implementation and transformation, and promoting digital transformation and intelligent upgrading across industries.

## Q: What is a Pipeline? What is a Module? What is the relationship between them?

A: In PaddleX, a module is defined as the smallest unit that implements basic functions, meaning each module undertakes a specific task, such as text detection. Within this framework, a pipeline is the actual functionality achieved by one or more modules working together, often forming more complex application scenarios, such as Optical Character Recognition (OCR) technology. Therefore, the relationship between modules and pipelines can be understood as the relationship between basics and applications. Modules, as the smallest units, provide the foundation for construction, while pipelines demonstrate the practical application effects of these foundational modules after reasonable combination and configuration. This design approach allows users to flexibly select and combine different modules to achieve the functions they need, significantly enhancing development flexibility and efficiency. The official pipelines also support users with high-performance inference, serving deployment, and other deployment capabilities.

## Q: How to choose between the Wheel package installation mode and the plugin installation mode?

A: If your application scenario in using PaddleX mainly focuses on model inference and integration, we highly recommend choosing a more convenient and lightweight installation method, namely the Wheel package installation mode. This installation mode aims to provide users with a quick and simple installation experience, especially suitable for scenarios requiring rapid deployment and integration of models. Installing with Wheel packages can significantly reduce the complexity of the installation process, avoid unnecessary configuration issues, and allow developers to focus more time and effort on the practical application and optimization of models. Whether you are a novice or an experienced developer, this lightweight installation method will greatly facilitate your workflow. Therefore, when performing model inference and integration, choosing the Wheel package installation mode will undoubtedly make your entire development process more efficient and smooth.

## Q: What is the difference between PaddleX and Baidu AIStudio Community's Zero-Code Pipeline?

A: Baidu AIStudio Community's Zero-Code Pipeline is the cloud-based carrier of PaddleX, with its underlying code consistent with PaddleX, and can be considered as a cloud-based PaddleX. The design philosophy of Baidu AIStudio Community's Zero-Code Pipeline is to enable users to quickly build and deploy model applications without needing to delve deeply into programming and algorithm knowledge. On this basis, Baidu AIStudio Community's Zero-Code Pipeline also provides many special pipelines, such as training high-precision models with a small number of samples and solving complex time-series problems using multi-model fusion schemes. PaddleX, on the other hand, is a local development tool that provides users with powerful functions supporting more in-depth secondary development. This means developers can flexibly adjust and expand based on PaddleX to create solutions that better fit specific application scenarios. Additionally, PaddleX offers a rich set of model interfaces, supporting users in freely combining models for use.

## <b>Q: How to continue training from a previously trained model?</b>

A: To resume training from a saved checkpoint, set the `pretrain_weights` parameter to the path of the previously saved model when calling the `train` interface.

## <b>Q: What are the differences between various models saved by PaddleX and how to distinguish them?</b>

A: The purposes of different types of models are as follows:

1. **Normally Trained Saved Model**: Suitable for loading predictions, serving as pre-training weights, or exporting deployment models.
3. **Exported Deployment Model**: Designed for server-side deployment and cannot be used as pre-training weights.

To distinguish between these models, check the `status` field in the `model.yml` file within the model directory:
- `Normal`: Normally trained model
- `Infer`: Deployment model

## <b>Q: Every time I start a new training session, it tries to re-download the pretrained models. Can this be avoided?</b>

A: Yes, you have two options:

1. Manually manage the models as described previously.
2. Set a global cache path for pretrained models, e.g., `paddlex.pretrain_dir='/usrname/paddlex'`. Models already downloaded to this directory will not be redownloaded.

## Q: When I encounter problems while using PaddleX, how should I provide feedback?

A: Welcome to the [Discussion Area](https://github.com/PaddlePaddle/PaddleX/discussions) to communicate with a vast number of developers! If you find errors or deficiencies in PaddleX, you are also welcome to [submit an issue](https://github.com/PaddlePaddle/PaddleX/issues), and our on-duty team members will respond to your questions as soon as possible.
