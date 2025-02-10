---
comments: true
---

# 人脸检测模块使用教程

## 一、概述
人脸检测任务是目标检测中的一项基本任务，旨在从输入图像中自动识别并定位人脸的位置和大小。它是人脸识别、人脸分析等后续任务的前提和基础。人脸检测任务通过构建深度神经网络模型，学习人脸的特征表示，实现高效准确的人脸检测。

## 二、支持模型列表

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>AP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>BlazeFace</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">训练模型</a></td>
<td>15.4</td>
<td>60.34 / 54.76</td>
<td>84.18 / 84.18</td>
<td>0.447</td>
<td>轻量高效的人脸检测模型</td>
</tr>
<tr>
<td>BlazeFace-FPN-SSH</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace-FPN-SSH_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">训练模型</a></td>
<td>18.7</td>
<td>69.29 / 63.42</td>
<td>86.96 / 86.96</td>
<td>0.606</td>
<td>BlazeFace的改进模型，增加FPN和SSH结构</td>
</tr>
<tr>
<td>PicoDet_LCNet_x2_5_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">训练模型</a></td>
<td>31.4</td>
<td>35.37 / 12.88</td>
<td>126.24 / 126.24</td>
<td>28.9</td>
<td>基于PicoDet_LCNet_x2_5的人脸检测模型</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">训练模型</a></td>
<td>36.1</td>
<td>22.54 / 8.33</td>
<td>138.67 / 138.67</td>
<td>26.5</td>
<td>基于PP-YOLOE_plus-S的人脸检测模型</td>
</tr>
</tbody>
</table>
<p>注：以上精度指标是在 COCO 格式的 WIDER-FACE 验证集上，以640
*640作为输入尺寸评估得到的。所有模型 GPU 推理耗时基于 NVIDIA V100 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz，精度类型为 FP32。</p>


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成whl包的安装后，几行代码即可完成人脸检测模块的推理，可以任意切换该模块下的模型，您也可以将人脸检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png)到本地。

```python
from paddlex import create_model

model_name = "PicoDet_LCNet_x2_5_face"

model = create_model(model_name)
output = model.predict("face_detection.png", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': {'input_path': 'face_detection.png', 'boxes': [{'cls_id': 0, 'label': 'face', 'score': 0.748367965221405, 'coordinate': [586.83154296875, 342.7668151855469, 595.56005859375, 352.7322692871094]}, {'cls_id': 0, 'label': 'face', 'score': 0.7040695548057556, 'coordinate': [679.3336181640625, 252.70413208007812, 687.8116455078125, 264.118896484375]}, {'cls_id': 0, 'label': 'face', 'score': 0.7013024687767029, 'coordinate': [655.5321044921875, 359.2837219238281, 664.028564453125, 370.1717224121094]}, {'cls_id': 0, 'label': 'face', 'score': 0.6968276500701904, 'coordinate': [385.4735107421875, 106.36427307128906, 392.57501220703125, 115.34565734863281]}, {'cls_id': 0, 'label': 'face', 'score': 0.6547412872314453, 'coordinate': [460.8025207519531, 304.8067626953125, 468.662841796875, 315.1859130859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.654322624206543, 'coordinate': [641.4616088867188, 9.26927661895752, 647.674560546875, 17.80522918701172]}, {'cls_id': 0, 'label': 'face', 'score': 0.6522411704063416, 'coordinate': [804.1121826171875, 612.6964721679688, 813.2249755859375, 624.7404174804688]}, {'cls_id': 0, 'label': 'face', 'score': 0.6494446992874146, 'coordinate': [567.8344116210938, 254.95977783203125, 576.0180053710938, 265.2414855957031]}, {'cls_id': 0, 'label': 'face', 'score': 0.647239089012146, 'coordinate': [784.8236083984375, 69.3985824584961, 791.87060546875, 78.83570861816406]}, {'cls_id': 0, 'label': 'face', 'score': 0.645237922668457, 'coordinate': [461.0616760253906, 170.65554809570312, 468.1227722167969, 180.4543914794922]}, {'cls_id': 0, 'label': 'face', 'score': 0.6428831219673157, 'coordinate': [596.9057006835938, 145.5034942626953, 603.8390502929688, 155.76255798339844]}, {'cls_id': 0, 'label': 'face', 'score': 0.635290265083313, 'coordinate': [978.6275024414062, 632.8814086914062, 989.2858276367188, 645.9703369140625]}, {'cls_id': 0, 'label': 'face', 'score': 0.6351185441017151, 'coordinate': [687.6617431640625, 138.90045166015625, 694.2169799804688, 148.66720581054688]}, {'cls_id': 0, 'label': 'face', 'score': 0.6283911466598511, 'coordinate': [423.2082214355469, 127.0450439453125, 429.9551696777344, 136.3946533203125]}, {'cls_id': 0, 'label': 'face', 'score': 0.6244862079620361, 'coordinate': [581.82275390625, 200.42108154296875, 588.7484741210938, 209.8951416015625]}, {'cls_id': 0, 'label': 'face', 'score': 0.6221384406089783, 'coordinate': [370.1514587402344, 168.1388702392578, 378.11297607421875, 178.57049560546875]}, {'cls_id': 0, 'label': 'face', 'score': 0.6171290278434753, 'coordinate': [655.06396484375, 190.71421813964844, 662.4423217773438, 200.9834442138672]}, {'cls_id': 0, 'label': 'face', 'score': 0.6137390732765198, 'coordinate': [398.8143310546875, 192.92584228515625, 406.77587890625, 203.5181121826172]}, {'cls_id': 0, 'label': 'face', 'score': 0.6107531189918518, 'coordinate': [426.2248229980469, 174.40940856933594, 433.2980651855469, 184.15167236328125]}, {'cls_id': 0, 'label': 'face', 'score': 0.6092986464500427, 'coordinate': [324.9934997558594, 117.24430847167969, 331.8951110839844, 126.23627471923828]}, {'cls_id': 0, 'label': 'face', 'score': 0.607699990272522, 'coordinate': [544.6709594726562, 153.3218536376953, 551.3650512695312, 163.32049560546875]}, {'cls_id': 0, 'label': 'face', 'score': 0.6042397022247314, 'coordinate': [513.06103515625, 187.57003784179688, 519.810546875, 196.89759826660156]}, {'cls_id': 0, 'label': 'face', 'score': 0.6038359999656677, 'coordinate': [517.808837890625, 325.5520935058594, 525.3433227539062, 335.09844970703125]}, {'cls_id': 0, 'label': 'face', 'score': 0.60162353515625, 'coordinate': [605.2512817382812, 9.034326553344727, 611.43359375, 17.63986587524414]}, {'cls_id': 0, 'label': 'face', 'score': 0.6000550985336304, 'coordinate': [308.621337890625, 161.19313049316406, 316.0295104980469, 171.12921142578125]}, {'cls_id': 0, 'label': 'face', 'score': 0.5962246656417847, 'coordinate': [750.8837280273438, 595.826416015625, 761.5985107421875, 609.9155883789062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5959786176681519, 'coordinate': [815.2214965820312, 119.83965301513672, 822.11572265625, 128.9686279296875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5952416062355042, 'coordinate': [129.22276306152344, 341.8433532714844, 140.06057739257812, 356.851318359375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5921688079833984, 'coordinate': [830.36083984375, 231.48342895507812, 838.8031616210938, 242.44595336914062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5867078304290771, 'coordinate': [1011.5479125976562, 170.10679626464844, 1018.60205078125, 179.95533752441406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5841799378395081, 'coordinate': [452.0153503417969, 143.87330627441406, 458.7742004394531, 153.7798614501953]}, {'cls_id': 0, 'label': 'face', 'score': 0.5795201659202576, 'coordinate': [942.02880859375, 104.56780242919922, 948.6239624023438, 113.49226379394531]}, {'cls_id': 0, 'label': 'face', 'score': 0.578009843826294, 'coordinate': [665.4860229492188, 12.127445220947266, 671.739013671875, 19.92763328552246]}, {'cls_id': 0, 'label': 'face', 'score': 0.575183629989624, 'coordinate': [548.813232421875, 205.44232177734375, 555.7802734375, 214.8798065185547]}, {'cls_id': 0, 'label': 'face', 'score': 0.5749289393424988, 'coordinate': [525.3703002929688, 37.35319137573242, 532.8809204101562, 48.29341125488281]}, {'cls_id': 0, 'label': 'face', 'score': 0.5741029977798462, 'coordinate': [267.226318359375, 203.7793731689453, 276.1531677246094, 215.31790161132812]}, {'cls_id': 0, 'label': 'face', 'score': 0.5717172622680664, 'coordinate': [487.4885559082031, 133.73777770996094, 494.1946105957031, 143.52841186523438]}, {'cls_id': 0, 'label': 'face', 'score': 0.5716090202331543, 'coordinate': [847.9454956054688, 120.08977508544922, 855.2705078125, 129.53871154785156]}, {'cls_id': 0, 'label': 'face', 'score': 0.5712835788726807, 'coordinate': [344.761474609375, 175.31829833984375, 352.2358093261719, 185.17408752441406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5700110793113708, 'coordinate': [265.0657958984375, 170.31253051757812, 271.9244079589844, 179.15249633789062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5698822736740112, 'coordinate': [574.304931640625, 118.77815246582031, 581.50439453125, 129.45875549316406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5661887526512146, 'coordinate': [327.7814025878906, 151.88540649414062, 336.2471008300781, 162.232421875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5657126903533936, 'coordinate': [766.1329345703125, 380.17059326171875, 773.3143920898438, 389.7149963378906]}, {'cls_id': 0, 'label': 'face', 'score': 0.5656450390815735, 'coordinate': [262.93048095703125, 135.5059814453125, 270.763916015625, 145.3970947265625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5647367238998413, 'coordinate': [993.9485473632812, 194.26812744140625, 1000.8638305664062, 203.5723419189453]}, {'cls_id': 0, 'label': 'face', 'score': 0.5644330382347107, 'coordinate': [464.95098876953125, 143.08924865722656, 472.000732421875, 153.5122833251953]}, {'cls_id': 0, 'label': 'face', 'score': 0.5629076957702637, 'coordinate': [434.9204406738281, 149.97059631347656, 441.84368896484375, 159.5159454345703]}, {'cls_id': 0, 'label': 'face', 'score': 0.5591433644294739, 'coordinate': [484.9643859863281, 216.93482971191406, 491.9638671875, 226.16673278808594]}, {'cls_id': 0, 'label': 'face', 'score': 0.5577755570411682, 'coordinate': [697.8162231445312, 467.0403137207031, 707.13623046875, 481.5167541503906]}, {'cls_id': 0, 'label': 'face', 'score': 0.5571652054786682, 'coordinate': [575.9535522460938, 173.58697509765625, 582.6239013671875, 183.01803588867188]}, {'cls_id': 0, 'label': 'face', 'score': 0.5568857789039612, 'coordinate': [854.416259765625, 400.8706970214844, 862.5512084960938, 411.5679016113281]}, {'cls_id': 0, 'label': 'face', 'score': 0.5568552017211914, 'coordinate': [726.2380981445312, 134.23175048828125, 732.9859619140625, 143.24359130859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5552074909210205, 'coordinate': [946.8074951171875, 150.908203125, 953.7364501953125, 160.1160888671875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5540658831596375, 'coordinate': [638.5432739257812, 154.90652465820312, 645.052490234375, 164.36062622070312]}, {'cls_id': 0, 'label': 'face', 'score': 0.5524702668190002, 'coordinate': [908.3207397460938, 571.7152099609375, 917.85498046875, 582.7459716796875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5519255995750427, 'coordinate': [392.22674560546875, 161.07704162597656, 399.60980224609375, 171.21505737304688]}, {'cls_id': 0, 'label': 'face', 'score': 0.5512804388999939, 'coordinate': [290.897705078125, 167.27468872070312, 298.8089599609375, 177.54385375976562]}, {'cls_id': 0, 'label': 'face', 'score': 0.549347996711731, 'coordinate': [741.4368286132812, 96.00164794921875, 748.0814208984375, 104.84812927246094]}, {'cls_id': 0, 'label': 'face', 'score': 0.5493444800376892, 'coordinate': [627.7223510742188, 111.2412338256836, 634.6829833984375, 120.89079284667969]}, {'cls_id': 0, 'label': 'face', 'score': 0.5484412908554077, 'coordinate': [942.8670654296875, 169.956298828125, 949.753662109375, 179.58168029785156]}, {'cls_id': 0, 'label': 'face', 'score': 0.5433928370475769, 'coordinate': [495.8935546875, 161.2371368408203, 502.8511657714844, 171.831787109375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5411538481712341, 'coordinate': [966.8755493164062, 89.61585235595703, 974.01708984375, 99.20806884765625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5397874116897583, 'coordinate': [911.2400512695312, 85.85540008544922, 918.2730712890625, 94.37992095947266]}, {'cls_id': 0, 'label': 'face', 'score': 0.5373152494430542, 'coordinate': [904.8760375976562, 101.83809661865234, 912.178466796875, 111.14995574951172]}, {'cls_id': 0, 'label': 'face', 'score': 0.5363165736198425, 'coordinate': [812.86767578125, 183.46951293945312, 819.8478393554688, 192.6529083251953]}, {'cls_id': 0, 'label': 'face', 'score': 0.5351170301437378, 'coordinate': [974.45849609375, 153.04583740234375, 982.1405029296875, 163.85182189941406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5350357890129089, 'coordinate': [746.5970458984375, 113.35935974121094, 753.0759887695312, 122.73274230957031]}, {'cls_id': 0, 'label': 'face', 'score': 0.5328549146652222, 'coordinate': [881.665771484375, 157.66879272460938, 888.5724487304688, 166.84524536132812]}, {'cls_id': 0, 'label': 'face', 'score': 0.532268762588501, 'coordinate': [509.4955749511719, 128.1408233642578, 516.4570922851562, 137.71487426757812]}, {'cls_id': 0, 'label': 'face', 'score': 0.5312089920043945, 'coordinate': [748.0822143554688, 14.927801132202148, 754.0698852539062, 23.060955047607422]}, {'cls_id': 0, 'label': 'face', 'score': 0.5301514863967896, 'coordinate': [698.60107421875, 592.9287719726562, 709.08740234375, 608.2547607421875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5291982889175415, 'coordinate': [853.537353515625, 497.73388671875, 862.7291870117188, 510.3872375488281]}, {'cls_id': 0, 'label': 'face', 'score': 0.5284016132354736, 'coordinate': [877.9340209960938, 195.67881774902344, 884.9655151367188, 205.39552307128906]}, {'cls_id': 0, 'label': 'face', 'score': 0.524852991104126, 'coordinate': [443.56048583984375, 125.34044647216797, 450.53216552734375, 134.87721252441406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5237076878547668, 'coordinate': [864.43798828125, 98.259765625, 871.5907592773438, 106.94507598876953]}, {'cls_id': 0, 'label': 'face', 'score': 0.523581862449646, 'coordinate': [895.39599609375, 203.71728515625, 902.3679809570312, 213.05873107910156]}, {'cls_id': 0, 'label': 'face', 'score': 0.5232769250869751, 'coordinate': [508.1500549316406, 94.16983032226562, 515.2929077148438, 104.36737060546875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5197194218635559, 'coordinate': [964.1079711914062, 268.97369384765625, 971.7310180664062, 279.86053466796875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5187004804611206, 'coordinate': [705.9102783203125, 267.20501708984375, 713.0789184570312, 276.50921630859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5185328722000122, 'coordinate': [188.63758850097656, 192.71762084960938, 195.5178985595703, 201.68702697753906]}, {'cls_id': 0, 'label': 'face', 'score': 0.5183883905410767, 'coordinate': [841.8475952148438, 152.13719177246094, 848.8064575195312, 162.1306610107422]}, {'cls_id': 0, 'label': 'face', 'score': 0.5182327032089233, 'coordinate': [618.0513305664062, 54.13559341430664, 625.4635009765625, 63.254356384277344]}, {'cls_id': 0, 'label': 'face', 'score': 0.5181956887245178, 'coordinate': [90.88504791259766, 103.90827178955078, 96.98222351074219, 111.67485046386719]}, {'cls_id': 0, 'label': 'face', 'score': 0.5179163813591003, 'coordinate': [696.462158203125, 191.00534057617188, 704.2958984375, 200.7041015625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5152389407157898, 'coordinate': [1004.4434814453125, 142.3948974609375, 1011.4993286132812, 152.9167022705078]}, {'cls_id': 0, 'label': 'face', 'score': 0.5132371783256531, 'coordinate': [824.3988647460938, 104.93993377685547, 831.6600341796875, 114.49164581298828]}, {'cls_id': 0, 'label': 'face', 'score': 0.512485682964325, 'coordinate': [182.48626708984375, 241.70651245117188, 191.400390625, 253.36883544921875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5107918381690979, 'coordinate': [677.3311157226562, 23.615076065063477, 683.8475341796875, 32.1396369934082]}, {'cls_id': 0, 'label': 'face', 'score': 0.5083979964256287, 'coordinate': [919.97705078125, 216.38670349121094, 927.0363159179688, 225.7605438232422]}, {'cls_id': 0, 'label': 'face', 'score': 0.508198618888855, 'coordinate': [788.8171997070312, 92.79387664794922, 795.2500610351562, 101.81549835205078]}, {'cls_id': 0, 'label': 'face', 'score': 0.5081363320350647, 'coordinate': [649.3057250976562, 137.01023864746094, 655.3905029296875, 145.69985961914062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5073381066322327, 'coordinate': [935.124755859375, 229.1543426513672, 941.9274291992188, 238.734130859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5042576789855957, 'coordinate': [113.26185607910156, 232.80154418945312, 121.303955078125, 242.5761260986328]}, {'cls_id': 0, 'label': 'face', 'score': 0.5031769275665283, 'coordinate': [676.889892578125, 168.4464111328125, 683.51904296875, 177.55438232421875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5023666620254517, 'coordinate': [962.6386108398438, 188.63504028320312, 969.4259033203125, 198.957275390625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5021270513534546, 'coordinate': [403.39642333984375, 133.8506622314453, 411.1600341796875, 144.05316162109375]}]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `boxes`：预测的目标框信息，一个字典列表。每个字典包含以下信息：
  - `cls_id`：类别ID，一个整数
  - `label`：类别标签，一个字符串
  - `score`：目标框置信度，一个浮点数
  - `coordinate`：目标框坐标，一个列表[xmin, ymin, xmax, ymax]

可视化图像如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/face_det/face_detection_res.png"/>

相关方法、参数等说明如下：

* `create_model`实例化人脸检测模型（此处以`PicoDet_LCNet_x2_5_face`为例），具体说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>模型名称</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>输入图像大小；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>int/list/None</code></td>
<td>
<ul>
<li><b>int</b>, 如 640 , 表示将输入图像resize到640x640大小</li>
<li><b>列表</b>, 如 [640, 512] , 表示将输入图像resize到宽为640，高为512大小</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>float/None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用人脸检测模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`、`batch_size`和`threshold`，具体说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
<li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
<li><b>URL链接</b>，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">示例</a></li>
<li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
<li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>任意整数</td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值；如果不指定，将默认使用 <code>creat_model</code> 指定的<code>threshold</code> 参数，如果<code>creat_model</code> 也没有指定， 则默认使用PaddleX官方模型配置</td>
<td><code>float</code></td>
<td>无</td>
<td>无</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">获取格式为<code>dict</code>的可视化图像</td>
</tr>
</table>

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。
## 四、二次开发
如果你追求更高精度的现有模型，可以使用PaddleX的二次开发能力，开发更好的人脸检测模型。在使用PaddleX开发人脸检测模型之前，请务必安装PaddleX的PaddleDetection插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX为每一个模块都提供了demo数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX目标检测任务模块数据标注教程](../../../data_annotations/cv_modules/object_detection.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/widerface_coco_examples.tar -P ./dataset
tar -xf ./dataset/widerface_coco_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 500,
    "train_sample_paths": [
      "check_dataset/demo_img/0--Parade/0_Parade_marchingband_1_849.jpg",
      "check_dataset/demo_img/0--Parade/0_Parade_Parade_0_904.jpg",
      "check_dataset/demo_img/0--Parade/0_Parade_marchingband_1_799.jpg"
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_384.jpg",
      "check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_538.jpg",
      "check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_429.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
<<<<<<< HEAD
  "dataset_path": "./dataset/example_data/widerface_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
=======
  &quot;dataset_path&quot;: &quot;widerface_coco_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
>>>>>>> modify_pipeline_and_module_docs
}
</code></pre>
<p>上述校验结果中，<code>check_pass</code> 为 <code>True</code> 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 1；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 500；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 100；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/face_det/01.png"/></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>
<p><b>（1）数据集格式转换</b></p>
<p>人脸检测不支持数据格式转换。</p>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证与 <code>val_percent</code> 的值之和为100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处PicoDet_LCNet_x2_5_face的训练为例：

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet_LCNet_x2_5_face.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>
<p>在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
</li>
<li>
<p><code>train_result.json</code>：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；</p>
</li>
<li><code>train.log</code>：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；</li>
<li><code>config.yaml</code>：训练配置文件，记录了本次训练的超参数的配置；</li>
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet_LCNet_x2_5_face.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=``./output/best_model/best_model/model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；</p></details>

### <b>4.4 模型推理</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测。在PaddleX中实现模型推理预测可以通过两种方式：命令行和wheel 包。

#### 4.4.1 模型推理
* 通过命令行的方式进行推理预测，只需如下一条命令，运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png)到本地。
```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="face_detection.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet_LCNet_x2_5_face.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

人脸检测模块可以集成的PaddleX产线有[<b>人脸识别</b>](../../../pipeline_usage/tutorials/cv_pipelines/face_recognition.md)，只需要替换模型路径即可完成相关产线的人脸检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到人脸检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

您也可以利用 PaddleX 高性能推理插件来优化您模型的推理过程，进一步提升效率，详细的流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。
