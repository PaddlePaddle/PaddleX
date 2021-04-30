# 模型配置文件说明

模型配置文件主要用于描述模型的一些信息以及前后处理的一些操作，其大致结构如下。自定义模型的配置文件参照它即可：

```yaml
model_format: Paddle
toolkit: PaddleClas
toolkit_version: unknown
input_tensor_name: inputs
transforms:
  BGR2RGB: ~
  ResizeByShort:
    target_size: 256
    interp: 1
    use_scale: false
  CenterCrop:
    width: 224
    height: 224
  Convert:
    dtype: float
  Normalize:
    mean:
      - 0.485
      - 0.456
      - 0.406
    std:
      - 0.229
      - 0.224
      - 0.225
  Permute: ~
labels:
  - kit_fox
  - English_setter
  - Siberian_husky
```



## 配置文件字段解释

**model_format**                     
模型导出时所用框架，目前默认都为 `Paddle`。以后支持其他框架部署时，需要填写框架名。

**toolkit**  
模型导出时所用的套件名，现在支持 PaddleX、PaddleDetection、PaddleSeg 和 PaddleClas。如果是自定义框架解析，填写自己定义的框架名字即可。

**toolkit_version**  
模型导出时所用套件的版本，当前支持一个版本，默认为 unknown。以后支持多版本时，需要填写对应的版本号。

**input_tensor_name**  
模型输入的tensor名。PaddleClas套件默认为inputs，支持的其他套件不用填，自定义模型按需填。

**labels**  
标签列表。如果后处理要用到就需要填写

**transforms**（必要字段）  
里边是模型预处理时相关操作，一般主要修改这里，组装自己模型的预处理操作。部署程序会根据transforms里描写的操作，对图片按顺序执行相应的操作。  
transforms里是一个字典，每一个操作为一个键值对。key是操作名，value是操作的参数。  
例如下面的例子就是将一个图片数据从BGR转到RGB，然后将数据格式转为float类型，最后对图片进行归一化。

```yaml
transforms:
  BGR2RGB: ~             #BGR2RGB为操作名，表示BGR2RGB操作；~ 代表空，表示无参数
  Convert:               #操作名，表示Convert数据类型转换操作
    dtype: float         #dtype参数，它的值为 float
  Normalize:             #操作名，表示Normalize归一化操作
    mean:                #参数名， mean参数
      - 0.485            #mean参数,参数值为列表，为[0.485,0.456,0.406]
      - 0.456
      - 0.406
    std:                 #std参数，参数值为列表，为[0.229, 0.224, 0.225]
      - 0.229
      - 0.224
      - 0.225
```

transform里具体的操作说明如下



## 所有操作说明

**BGR2RGB**  
作用：将BGR格式图片数据转为RGB格式。  
参数:

```
    无输入参数。c++ yaml文件 ~ 代表空，也可以使用 null(例如BGR2RGB:  ~ 、BGR2RGB:  null)。
```
详细描述: 其他预处理操作和模型推理只能处理RGB格式，如果输入数据为BGR格式就需要在预处理的最开始用BGR2RGB。例如：用opencv的imread读取图片默认格式为BGR  



**RGB2BGR**  
作用：将RGB格式图片数据转为GBR格式。 
参数:  
```
    无输入参数
```
详细描述: 有时我们想要最终的结果为BGR格式，那可以在后处理的最后使用这个操作将RGB格式的结果转为BGR。



**Convert**  
作用：转换图片的数据格式  
参数: 
```
    dtype        string类型,转换后的数据类型。当前只支持float
```
详细描述: 用opencv读取图片的格式为uint8，进行浮点数计算(比如Normalize操作)之前，为了减少计算误差可以先将数据转为float。



**Permute**  
作用：改变图片的排列格式，从hwc格式转为chw格式  
参数:  
```
    无输入参数
```
详细描述: Paddle推理引擎默认处理的图片格式为chw。 用opencv读取图片的格式为hwc,在进入推理前需要用Permute转为chw格式。



**Resize**  
作用：调整图片大小  
参数: 
```
    height        int类型,调整后图片的高
    width         int类型,调整后图片的宽
    interp        int类型，图像缩放的差值算法类型，默认为1(线性插值)。对应opencv::resize中的interpolation参数
    use_scale  bool类型，是否使用比列缩放到指定大小，默认为true。
```
详细描述: 等比例缩放图片的大小，可用于将各尺寸图片都调整为模型输入需要的图片大小。



**ResizeByShort**  
作用：按图片最短边跟目标大小的比例，调整图片大小。 
参数: 
```
    target_size    int类型，调整后图片最短边大小
    interp            int类型，图像缩放的差值算法类型，默认为1(线性插值)。对应opencv::resize中的interpolation参数
    max_size       int类型,   图片的最大长度
    use_scale      bool类型，是否将target_size换算成系数来缩放,默认为true。
```
详细描述:  
例如原始图片的高和宽为(100,200)，目标大小为300，则转换后图片的高和宽为(300,  300/100200)=>(300, 600)。  
如果设置了图片的最大长度为400，由于600大于400，则图片最终调整为 (400/200\*100, 400/200*200)=>(200, 400)  
如果设置了图片的最大长度为800, 300和600都小于800，则图片最终还是调整为(300, 600)



**ResizeByLong**  
作用：按图片最长边跟目标大小的比例，调整图片大小.  
参数: 
```
    target_size    int类型，调整后图片最长边大小
    interp             int类型，图像缩放的差值算法类型，默认为1(线性插值)。对应opencv::resize中的interpolation参数
    max_size       int类型,   图片的最大长度
    use_scale      bool类型，是否使用比列缩放到指定大小，默认为true。
```
详细描述:  
例如原始图片的高和宽为(100,200)，目标大小为300，则转换后图片的高和宽为(300/200*100,  300)=>(150, 300)。  
如果设置了图片的最大长度为200，由于300大于200，则图片最终调整为 (200/200\*100, 200/200\*200)=>(100, 200)  
如果设置了图片的最大长度为400, 150和300都小于400，则图片最终还是调整为(150, 300)



**Normalize**  
作用：将图片进行归一化.  
参数: 
```
    mean        float数组，元素个数为channel数。归一化公示中的mean
    std             float数组，元素个数为channel数。归一化公示中的std
    min_val     float数组，元素个数为channel数。缩放公式中的min_val
    max_val     float数组，元素个数为channel数。缩放公式中的max_val
    is_scale     bool类型，是否进行缩放，默认为false。
```
详细描述: 默认归一化公式为 (input[channel] - mean[channel]) / std[channel]， 如果is_scale为true会先进行缩放  (input[channel] - min_val[channel]) / (max_val[channel] - min_val[channel]) 然后再进行归一化。



**Padding**  
作用：将图片填充到某个固定的大小  
参数: 
```
    height        int类型,  填充后图片的高
    width         int类型,  填充后图片的宽
    stride         int类型，默认值为1. 填充后必须为stride的倍数，主要用于fpn结构
    im_padding_value  float数组，元素个数为channel数，默认为0.0。指定填充的值
```
详细描述:  用某个值( im_padding_value)将图片填充到固定的大小。如果是fpn结构等对输出大小有倍数要求，则需要设定 stride参数



**CenterCrop**  
作用：取图片中心区域截取某个固定的大小图片  
参数: 
```
    height        int类型,截取图片的高
    width         int类型,截取图片的宽
```
详细描述: 图片的大小为(200,400)，设置高(height)为100,宽(width)为200。则会从中心区域(即高[50,150]的区间，宽[100,300]的区间)截取出大小为(100,200)的图片。

