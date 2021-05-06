# 部署API说明

本文档主要根据部署步骤，讲解用到的一些API、数据结构。

## 1. 创建模型对象

**std::shared_ptr\<Model>  ModelFactory::CreateObject(const std::string  &model_type)**

| 描述   | 根据套件类型，创建相应的套件对象并返回基类智能指针。所有推理相关的操作都在该对象中。                                                                                                                                                                                                                                         |
| ------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 参数   | model_type  套件类型，当前支持的套件为[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)、[PaddleX](https://github.com/PaddlePaddle/PaddleX)对应的model_type分别为 det、seg、clas、paddlex |
| 返回值 | 指向套件对象的父类指针                                                                                                                                                                                                                                                                                                       |
| 例子   | `std::shared_ptr\<PaddleDeploy::Model> model = PaddleDeploy::ModelFactory::CreateObject("det")` , 创建`DetModel`对象用于对`PaddleDetection`模型进行部署推理。                                                                                                                                                                |



## 2. 初始化

**bool Model::Init(const std::string& cfg_file)**

| 描述   | 初始化配置文件、模型前后处理。创建Model或者其子类套件对象后必须先调用它初始化，才能调推理接口。 |
| ------ | :---------------------------------------------------------------------------------------------- |
| 参数   | cfg_file   配置文件的路径                                                                       |
| 返回值 | 是否初始化成功                                                                                  |
| 例子   | 用上面创建的Model指针进行初始化， `model->Init("path/to/deploy.yaml")`                          |



**bool Model::PaddleEngineInit(**  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; const std::string&model_filename,  const std::string& params_filename,  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; bool use_gpu = false, int gpu_id = 0,  bool use_mkl = true);

| 描述   | 初始化Paddle 推理引擎,  创建Model或者其子类对象后必须先调用它初始化，才能调推理接口。                                                                                                                                                                                                                              |
| ------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 参数   | model_filename     paddle导出模型的模型文件<br />params_filename   paddle导出模型的参数文件<br />use_gpu                   是否使用GPU进行推理， 默认false。true为开启，false为关闭<br />gpu_id                    指定哪一张GPU卡进行推理，默认为0<br />use_mkl                  是否开启mkl进行加速， 默认为true |
| 返回值 | 是否初始化成功                                                                                                                                                                                                                                                                                                     |
| 例子   | 用上面创建的Model指针进行Paddle推理引擎初始化，开启GPU、mkl，使用第0张卡。 `model->PaddleEngineInit("path/to/model.pdmodel", "path/to/model.pdiparams", use_gpu = true)`                                                                                                                                           |



## 3.推理计算

 **virtual bool Predict(const std::vector\<cv::Mat>& imgs,  int thread_num = 1)**

**注意：**一般情况，调用此接口的到预测结果即可。

| 描述   | 进行预处理、模型推理、后处理计算。计算结果存于Model的成员变量 std::vector\<Result> results_中， 每次推理时，上一次的结果会被清空。 |
| ------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| 参数   | imgs                cv::Mat格式的图片数组<br />thread_num  OpenMP对batch并行加速的线程数                                           |
| 返回值 | 计算是否成功                                                                                                                       |
| 例子   | model->Predict(imgs) 输入图片为imgs数组，进行预处理、模型推理、后处理计算，结果保存在model->results_中。                           |



**virtual  bool PrePrecess(const std::vector\<cv::Mat>& imgs,**  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; std::vector\<DataBlob>***** inputs,  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; std::vector\<ShapeInfo>***** shape_infos,  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; int thread_num = 1)

| 描述   | 用于单独的预处理计算                                                                                                                                                                                                                                                                                                                 |
| ------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 参数   | imgs                cv::Mat格式的图片数组<br />inputs             原始图片数据经过预处理后得到的数据，作为模型推理时的输入。DataBlob类型的vector数组，DataBlob详细数据结构见下<br />shape_infos   存放预处理中每个操作后图片的形状， 可用于后处理还原图片。 ShapeInfo详细数据结构见下<br />thread_num  OpenMP对batch并行加速的线程数 |
| 返回值 | 计算是否成功                                                                                                                                                                                                                                                                                                                         |
| 例子   | model->PrePrecess(imgs, &inputs, &shape_infos) 输入图片为imgs数组，进行预处理计算。图片结果保存在inputs中，图片形状在shape_info中。                                                                                                                                                                                                  |



**virtual  void Infer(const std::vector\<DataBlob>& inputs,  std::vector\<DataBlob>* outputs)**

| 描述   | 用于单独的模型推理引擎计算                                                                                                             |
| ------ | :------------------------------------------------------------------------------------------------------------------------------------- |
| 参数   | inputs      模型推理的输入。DataBlob类型的vector数组，DataBlob详细数据结构见下<br />outputs   模型推理的输出，DataBlob类型的vector数组 |
| 返回值 | 计算是否成功                                                                                                                           |
| 例子   | model->Infer(inputs, &output) 输入数据为inputs数组，模型推理计算，结果保存在output中。                                                 |



**virtual bool PostPrecess(const std::vector\<DataBlob>& outputs,**  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; const std::vector\<ShapeInfo>& shape_infos,  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; int  thread_num = 1)

| 描述   | 用于单独的后处理计算                                                                                                                                                                                                   |
| ------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 参数   | outputs             模型推理的输出，DataBlob类型的vector数组<br />shape_infos     存放预处理中每个操作后图片的形状， 可用于后处理还原图片。 ShapeInfo类型的vector数组<br />thread_num    OpenMP对batch并行加速的线程数 |
| 返回值 | 计算是否成功                                                                                                                                                                                                           |
| 例子   | model->PostPrecess(outputs, shape_info ) 输入为output数组和shape_info，进行后处理计算，结果保存在model->results_中。                                                                                                   |



## 4. 结果获取和打印

**virtual void PrintResult()**

| 描述   | 遍历打印每一张图片的推理结果                         |
| ------ | :--------------------------------------------------- |
| 参数   | 无                                                   |
| 返回值 | 无                                                   |
| 例子   | model->PrintResult() 遍历打印model->results_中的结果 |



## 5.部署定义的数据结构

部署代码中各接口涉及到的自定义数据结构都定义在[output_struct.h](../../model_deploy/common/include/output_struct.h)，详细描述如下。

### 1.预测结果

**Result**

模型预测的最终结果用Result结构体保存，详细数据结果如下:

```c++
struct Result {
  std::string model_type;    //由哪种类型的模型计算得到的结果，只支持det, seg, clas
  union {                    //联合体，根据上边的类型对应的指针有效。 如model_type为det时，表示是Detection模型推理产生的结果，det_result指针有效。
    ClasResult* clas_result;
    DetResult* det_result;
    SegResult* seg_result;
  };
};
```

**DetResult**

保存det模型的预测结果

```c++
struct DetResult {
  // target boxes
  std::vector<Box> boxes;    // 存放检测出的所有box
};
```

**SegResult**

保存seg模型的预测结果

```c++
struct SegResult {
  Mask<uint8_t> label_map;   // 图片上每个像素的标签
  Mask<float> score_map;     // 图片上每个像素的打分
}
```

**ClasResult**

保存clas模型的推理结果

```c++
struct ClasResult {
  // target boxes
  int category_id;   //类别id
  std::string category;  //对应的类别名
  double score;      //打分 结果的置信度
}
```

**Box**

存放图片中框的信息，比如目标检测中识别出的目标框

```c++
struct Box {
  int category_id;      //类别id
  // category label this box belongs to
  std::string category;  //对应的类别名
  // confidence score
  float score;            //打分 结果的置信度
  std::vector<float> coordinate;  //框的左下角坐标(x、y)和宽、高，共四个值[xmin, ymin, width, heigt]
  Mask<uint8_t> mask;             //mask标记
};
```

### 2.推理接口涉及的输入

**DataBlob**

调用Model::Infer(推理引擎接口)进行模型推理引擎计算时，该数据结构作为输入参数。主要是保存图片的相关信息。

```c++
struct DataBlob {
  // data
  std::vector<char> data;  //char矩阵保存图片像素信息
  // data name
  std::string name;  //对应输入名
  // data shape
  std::vector<int> shape; // 图片形状大小
  /*
    data dtype
    0: FLOAT32
    1: INT64
    2: INT32
    3: UINT8
    */
  int dtype;   //图片的数据格式
  std::vector<std::vector<size_t>> lod; // 保存batch的信息
}
```

**ShapeInfo**

主要用于前后处理，保存预处理每一个操作后图片的形状信息

```c++
struct ShapeInfo {
  std::vector<std::vector<int>> shapes; // 预处理每一步操作后的形状大小
  std::vector<std::string> transforms;  // 顺序保存预处理操作名
}
```

