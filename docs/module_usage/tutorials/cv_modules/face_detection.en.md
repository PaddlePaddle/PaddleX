---
comments: true
---

# Face Detection Module Development Tutorial

## I. Overview
Face detection is a fundamental task in object detection, aiming to automatically identify and locate the position and size of faces in input images. It serves as the prerequisite and foundation for subsequent tasks such as face recognition and face analysis. Face detection accomplishes this by constructing deep neural network models that learn the feature representations of faces, enabling efficient and accurate face detection.

## II. Supported Model List


<table>
<thead>
<tr>
<th style="text-align: center;">Model</th><th>Model Download Link</th>
<th style="text-align: center;">AP (%)<br/>Easy/Medium/Hard</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th style="text-align: center;">Model Size (M)</th>
<th style="text-align: center;">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;">BlazeFace</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">77.7/73.4/49.5</td>
<td style="text-align: center;">60.34 / 54.76</td>
<td style="text-align: center;">84.18 / 84.18</td>
<td style="text-align: center;">0.447</td>
<td style="text-align: center;">A lightweight and efficient face detection model</td>
</tr>
<tr>
<td style="text-align: center;">BlazeFace-FPN-SSH</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace-FPN-SSH_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">83.2/80.5/60.5</td>
<td style="text-align: center;">69.29 / 63.42</td>
<td style="text-align: center;">86.96 / 86.96</td>
<td style="text-align: center;">0.606</td>
<td style="text-align: center;">An improved model of BlazeFace, incorporating FPN and SSH structures</td>
</tr>
<tr>
<td style="text-align: center;">PicoDet_LCNet_x2_5_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">93.7/90.7/68.1</td>
<td style="text-align: center;">35.37 / 12.88</td>
<td style="text-align: center;">126.24 / 126.24</td>
<td style="text-align: center;">28.9</td>
<td style="text-align: center;">Face Detection model based on PicoDet_LCNet_x2_5</td>
</tr>
<tr>
<td style="text-align: center;">PP-YOLOE_plus-S_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">93.9/91.8/79.8</td>
<td style="text-align: center;">22.54 / 8.33</td>
<td style="text-align: center;">138.67 / 138.67</td>
<td style="text-align: center;">26.5</td>
<td style="text-align: center;">Face Detection model based on PP-YOLOE_plus-S</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on the WIDER-FACE validation set with an input size of 640*640. GPU inference time is based on an NVIDIA V100 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz and FP32 precision.</b>


## III. Quick Integration  <a id="quick"> </a>

> ❗ Before quick integration, please install the PaddleX wheel package first. For details, please refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After completing the installation of the wheel package, you can perform inference for the face detection module with just a few lines of code. You can switch models under this module at will, and you can also integrate the model inference of the face detection module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png) to your local machine.

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


<details><summary>👉 <b>The result obtained after running is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'face_detection.png', 'boxes': [{'cls_id': 0, 'label': 'face', 'score': 0.748367965221405, 'coordinate': [586.83154296875, 342.7668151855469, 595.56005859375, 352.7322692871094]}, {'cls_id': 0, 'label': 'face', 'score': 0.7040695548057556, 'coordinate': [679.3336181640625, 252.70413208007812, 687.8116455078125, 264.118896484375]}, {'cls_id': 0, 'label': 'face', 'score': 0.7013024687767029, 'coordinate': [655.5321044921875, 359.2837219238281, 664.028564453125, 370.1717224121094]}, {'cls_id': 0, 'label': 'face', 'score': 0.6968276500701904, 'coordinate': [385.4735107421875, 106.36427307128906, 392.57501220703125, 115.34565734863281]}, {'cls_id': 0, 'label': 'face', 'score': 0.6547412872314453, 'coordinate': [460.8025207519531, 304.8067626953125, 468.662841796875, 315.1859130859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.654322624206543, 'coordinate': [641.4616088867188, 9.26927661895752, 647.674560546875, 17.80522918701172]}, {'cls_id': 0, 'label': 'face', 'score': 0.6522411704063416, 'coordinate': [804.1121826171875, 612.6964721679688, 813.2249755859375, 624.7404174804688]}, {'cls_id': 0, 'label': 'face', 'score': 0.6494446992874146, 'coordinate': [567.8344116210938, 254.95977783203125, 576.0180053710938, 265.2414855957031]}, {'cls_id': 0, 'label': 'face', 'score': 0.647239089012146, 'coordinate': [784.8236083984375, 69.3985824584961, 791.87060546875, 78.83570861816406]}, {'cls_id': 0, 'label': 'face', 'score': 0.645237922668457, 'coordinate': [461.0616760253906, 170.65554809570312, 468.1227722167969, 180.4543914794922]}, {'cls_id': 0, 'label': 'face', 'score': 0.6428831219673157, 'coordinate': [596.9057006835938, 145.5034942626953, 603.8390502929688, 155.76255798339844]}, {'cls_id': 0, 'label': 'face', 'score': 0.635290265083313, 'coordinate': [978.6275024414062, 632.8814086914062, 989.2858276367188, 645.9703369140625]}, {'cls_id': 0, 'label': 'face', 'score': 0.6351185441017151, 'coordinate': [687.6617431640625, 138.90045166015625, 694.2169799804688, 148.66720581054688]}, {'cls_id': 0, 'label': 'face', 'score': 0.6283911466598511, 'coordinate': [423.2082214355469, 127.0450439453125, 429.9551696777344, 136.3946533203125]}, {'cls_id': 0, 'label': 'face', 'score': 0.6244862079620361, 'coordinate': [581.82275390625, 200.42108154296875, 588.7484741210938, 209.8951416015625]}, {'cls_id': 0, 'label': 'face', 'score': 0.6221384406089783, 'coordinate': [370.1514587402344, 168.1388702392578, 378.11297607421875, 178.57049560546875]}, {'cls_id': 0, 'label': 'face', 'score': 0.6171290278434753, 'coordinate': [655.06396484375, 190.71421813964844, 662.4423217773438, 200.9834442138672]}, {'cls_id': 0, 'label': 'face', 'score': 0.6137390732765198, 'coordinate': [398.8143310546875, 192.92584228515625, 406.77587890625, 203.5181121826172]}, {'cls_id': 0, 'label': 'face', 'score': 0.6107531189918518, 'coordinate': [426.2248229980469, 174.40940856933594, 433.2980651855469, 184.15167236328125]}, {'cls_id': 0, 'label': 'face', 'score': 0.6092986464500427, 'coordinate': [324.9934997558594, 117.24430847167969, 331.8951110839844, 126.23627471923828]}, {'cls_id': 0, 'label': 'face', 'score': 0.607699990272522, 'coordinate': [544.6709594726562, 153.3218536376953, 551.3650512695312, 163.32049560546875]}, {'cls_id': 0, 'label': 'face', 'score': 0.6042397022247314, 'coordinate': [513.06103515625, 187.57003784179688, 519.810546875, 196.89759826660156]}, {'cls_id': 0, 'label': 'face', 'score': 0.6038359999656677, 'coordinate': [517.808837890625, 325.5520935058594, 525.3433227539062, 335.09844970703125]}, {'cls_id': 0, 'label': 'face', 'score': 0.60162353515625, 'coordinate': [605.2512817382812, 9.034326553344727, 611.43359375, 17.63986587524414]}, {'cls_id': 0, 'label': 'face', 'score': 0.6000550985336304, 'coordinate': [308.621337890625, 161.19313049316406, 316.0295104980469, 171.12921142578125]}, {'cls_id': 0, 'label': 'face', 'score': 0.5962246656417847, 'coordinate': [750.8837280273438, 595.826416015625, 761.5985107421875, 609.9155883789062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5959786176681519, 'coordinate': [815.2214965820312, 119.83965301513672, 822.11572265625, 128.9686279296875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5952416062355042, 'coordinate': [129.22276306152344, 341.8433532714844, 140.06057739257812, 356.851318359375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5921688079833984, 'coordinate': [830.36083984375, 231.48342895507812, 838.8031616210938, 242.44595336914062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5867078304290771, 'coordinate': [1011.5479125976562, 170.10679626464844, 1018.60205078125, 179.95533752441406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5841799378395081, 'coordinate': [452.0153503417969, 143.87330627441406, 458.7742004394531, 153.7798614501953]}, {'cls_id': 0, 'label': 'face', 'score': 0.5795201659202576, 'coordinate': [942.02880859375, 104.56780242919922, 948.6239624023438, 113.49226379394531]}, {'cls_id': 0, 'label': 'face', 'score': 0.578009843826294, 'coordinate': [665.4860229492188, 12.127445220947266, 671.739013671875, 19.92763328552246]}, {'cls_id': 0, 'label': 'face', 'score': 0.575183629989624, 'coordinate': [548.813232421875, 205.44232177734375, 555.7802734375, 214.8798065185547]}, {'cls_id': 0, 'label': 'face', 'score': 0.5749289393424988, 'coordinate': [525.3703002929688, 37.35319137573242, 532.8809204101562, 48.29341125488281]}, {'cls_id': 0, 'label': 'face', 'score': 0.5741029977798462, 'coordinate': [267.226318359375, 203.7793731689453, 276.1531677246094, 215.31790161132812]}, {'cls_id': 0, 'label': 'face', 'score': 0.5717172622680664, 'coordinate': [487.4885559082031, 133.73777770996094, 494.1946105957031, 143.52841186523438]}, {'cls_id': 0, 'label': 'face', 'score': 0.5716090202331543, 'coordinate': [847.9454956054688, 120.08977508544922, 855.2705078125, 129.53871154785156]}, {'cls_id': 0, 'label': 'face', 'score': 0.5712835788726807, 'coordinate': [344.761474609375, 175.31829833984375, 352.2358093261719, 185.17408752441406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5700110793113708, 'coordinate': [265.0657958984375, 170.31253051757812, 271.9244079589844, 179.15249633789062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5698822736740112, 'coordinate': [574.304931640625, 118.77815246582031, 581.50439453125, 129.45875549316406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5661887526512146, 'coordinate': [327.7814025878906, 151.88540649414062, 336.2471008300781, 162.232421875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5657126903533936, 'coordinate': [766.1329345703125, 380.17059326171875, 773.3143920898438, 389.7149963378906]}, {'cls_id': 0, 'label': 'face', 'score': 0.5656450390815735, 'coordinate': [262.93048095703125, 135.5059814453125, 270.763916015625, 145.3970947265625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5647367238998413, 'coordinate': [993.9485473632812, 194.26812744140625, 1000.8638305664062, 203.5723419189453]}, {'cls_id': 0, 'label': 'face', 'score': 0.5644330382347107, 'coordinate': [464.95098876953125, 143.08924865722656, 472.000732421875, 153.5122833251953]}, {'cls_id': 0, 'label': 'face', 'score': 0.5629076957702637, 'coordinate': [434.9204406738281, 149.97059631347656, 441.84368896484375, 159.5159454345703]}, {'cls_id': 0, 'label': 'face', 'score': 0.5591433644294739, 'coordinate': [484.9643859863281, 216.93482971191406, 491.9638671875, 226.16673278808594]}, {'cls_id': 0, 'label': 'face', 'score': 0.5577755570411682, 'coordinate': [697.8162231445312, 467.0403137207031, 707.13623046875, 481.5167541503906]}, {'cls_id': 0, 'label': 'face', 'score': 0.5571652054786682, 'coordinate': [575.9535522460938, 173.58697509765625, 582.6239013671875, 183.01803588867188]}, {'cls_id': 0, 'label': 'face', 'score': 0.5568857789039612, 'coordinate': [854.416259765625, 400.8706970214844, 862.5512084960938, 411.5679016113281]}, {'cls_id': 0, 'label': 'face', 'score': 0.5568552017211914, 'coordinate': [726.2380981445312, 134.23175048828125, 732.9859619140625, 143.24359130859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5552074909210205, 'coordinate': [946.8074951171875, 150.908203125, 953.7364501953125, 160.1160888671875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5540658831596375, 'coordinate': [638.5432739257812, 154.90652465820312, 645.052490234375, 164.36062622070312]}, {'cls_id': 0, 'label': 'face', 'score': 0.5524702668190002, 'coordinate': [908.3207397460938, 571.7152099609375, 917.85498046875, 582.7459716796875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5519255995750427, 'coordinate': [392.22674560546875, 161.07704162597656, 399.60980224609375, 171.21505737304688]}, {'cls_id': 0, 'label': 'face', 'score': 0.5512804388999939, 'coordinate': [290.897705078125, 167.27468872070312, 298.8089599609375, 177.54385375976562]}, {'cls_id': 0, 'label': 'face', 'score': 0.549347996711731, 'coordinate': [741.4368286132812, 96.00164794921875, 748.0814208984375, 104.84812927246094]}, {'cls_id': 0, 'label': 'face', 'score': 0.5493444800376892, 'coordinate': [627.7223510742188, 111.2412338256836, 634.6829833984375, 120.89079284667969]}, {'cls_id': 0, 'label': 'face', 'score': 0.5484412908554077, 'coordinate': [942.8670654296875, 169.956298828125, 949.753662109375, 179.58168029785156]}, {'cls_id': 0, 'label': 'face', 'score': 0.5433928370475769, 'coordinate': [495.8935546875, 161.2371368408203, 502.8511657714844, 171.831787109375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5411538481712341, 'coordinate': [966.8755493164062, 89.61585235595703, 974.01708984375, 99.20806884765625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5397874116897583, 'coordinate': [911.2400512695312, 85.85540008544922, 918.2730712890625, 94.37992095947266]}, {'cls_id': 0, 'label': 'face', 'score': 0.5373152494430542, 'coordinate': [904.8760375976562, 101.83809661865234, 912.178466796875, 111.14995574951172]}, {'cls_id': 0, 'label': 'face', 'score': 0.5363165736198425, 'coordinate': [812.86767578125, 183.46951293945312, 819.8478393554688, 192.6529083251953]}, {'cls_id': 0, 'label': 'face', 'score': 0.5351170301437378, 'coordinate': [974.45849609375, 153.04583740234375, 982.1405029296875, 163.85182189941406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5350357890129089, 'coordinate': [746.5970458984375, 113.35935974121094, 753.0759887695312, 122.73274230957031]}, {'cls_id': 0, 'label': 'face', 'score': 0.5328549146652222, 'coordinate': [881.665771484375, 157.66879272460938, 888.5724487304688, 166.84524536132812]}, {'cls_id': 0, 'label': 'face', 'score': 0.532268762588501, 'coordinate': [509.4955749511719, 128.1408233642578, 516.4570922851562, 137.71487426757812]}, {'cls_id': 0, 'label': 'face', 'score': 0.5312089920043945, 'coordinate': [748.0822143554688, 14.927801132202148, 754.0698852539062, 23.060955047607422]}, {'cls_id': 0, 'label': 'face', 'score': 0.5301514863967896, 'coordinate': [698.60107421875, 592.9287719726562, 709.08740234375, 608.2547607421875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5291982889175415, 'coordinate': [853.537353515625, 497.73388671875, 862.7291870117188, 510.3872375488281]}, {'cls_id': 0, 'label': 'face', 'score': 0.5284016132354736, 'coordinate': [877.9340209960938, 195.67881774902344, 884.9655151367188, 205.39552307128906]}, {'cls_id': 0, 'label': 'face', 'score': 0.524852991104126, 'coordinate': [443.56048583984375, 125.34044647216797, 450.53216552734375, 134.87721252441406]}, {'cls_id': 0, 'label': 'face', 'score': 0.5237076878547668, 'coordinate': [864.43798828125, 98.259765625, 871.5907592773438, 106.94507598876953]}, {'cls_id': 0, 'label': 'face', 'score': 0.523581862449646, 'coordinate': [895.39599609375, 203.71728515625, 902.3679809570312, 213.05873107910156]}, {'cls_id': 0, 'label': 'face', 'score': 0.5232769250869751, 'coordinate': [508.1500549316406, 94.16983032226562, 515.2929077148438, 104.36737060546875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5197194218635559, 'coordinate': [964.1079711914062, 268.97369384765625, 971.7310180664062, 279.86053466796875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5187004804611206, 'coordinate': [705.9102783203125, 267.20501708984375, 713.0789184570312, 276.50921630859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5185328722000122, 'coordinate': [188.63758850097656, 192.71762084960938, 195.5178985595703, 201.68702697753906]}, {'cls_id': 0, 'label': 'face', 'score': 0.5183883905410767, 'coordinate': [841.8475952148438, 152.13719177246094, 848.8064575195312, 162.1306610107422]}, {'cls_id': 0, 'label': 'face', 'score': 0.5182327032089233, 'coordinate': [618.0513305664062, 54.13559341430664, 625.4635009765625, 63.254356384277344]}, {'cls_id': 0, 'label': 'face', 'score': 0.5181956887245178, 'coordinate': [90.88504791259766, 103.90827178955078, 96.98222351074219, 111.67485046386719]}, {'cls_id': 0, 'label': 'face', 'score': 0.5179163813591003, 'coordinate': [696.462158203125, 191.00534057617188, 704.2958984375, 200.7041015625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5152389407157898, 'coordinate': [1004.4434814453125, 142.3948974609375, 1011.4993286132812, 152.9167022705078]}, {'cls_id': 0, 'label': 'face', 'score': 0.5132371783256531, 'coordinate': [824.3988647460938, 104.93993377685547, 831.6600341796875, 114.49164581298828]}, {'cls_id': 0, 'label': 'face', 'score': 0.512485682964325, 'coordinate': [182.48626708984375, 241.70651245117188, 191.400390625, 253.36883544921875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5107918381690979, 'coordinate': [677.3311157226562, 23.615076065063477, 683.8475341796875, 32.1396369934082]}, {'cls_id': 0, 'label': 'face', 'score': 0.5083979964256287, 'coordinate': [919.97705078125, 216.38670349121094, 927.0363159179688, 225.7605438232422]}, {'cls_id': 0, 'label': 'face', 'score': 0.508198618888855, 'coordinate': [788.8171997070312, 92.79387664794922, 795.2500610351562, 101.81549835205078]}, {'cls_id': 0, 'label': 'face', 'score': 0.5081363320350647, 'coordinate': [649.3057250976562, 137.01023864746094, 655.3905029296875, 145.69985961914062]}, {'cls_id': 0, 'label': 'face', 'score': 0.5073381066322327, 'coordinate': [935.124755859375, 229.1543426513672, 941.9274291992188, 238.734130859375]}, {'cls_id': 0, 'label': 'face', 'score': 0.5042576789855957, 'coordinate': [113.26185607910156, 232.80154418945312, 121.303955078125, 242.5761260986328]}, {'cls_id': 0, 'label': 'face', 'score': 0.5031769275665283, 'coordinate': [676.889892578125, 168.4464111328125, 683.51904296875, 177.55438232421875]}, {'cls_id': 0, 'label': 'face', 'score': 0.5023666620254517, 'coordinate': [962.6386108398438, 188.63504028320312, 969.4259033203125, 198.957275390625]}, {'cls_id': 0, 'label': 'face', 'score': 0.5021270513534546, 'coordinate': [403.39642333984375, 133.8506622314453, 411.1600341796875, 144.05316162109375]}]}}
```

The meanings of the parameters are as follows:
- `input_path`: The path of the input image to be predicted.
- `boxes`: Information of the predicted bounding boxes, a list of dictionaries. Each dictionary contains the following information:
  - `cls_id`: Class ID, an integer
  - `label`: Class label, a string
  - `score`: Confidence score of the bounding box, a float
  - `coordinate`: Coordinates of the bounding box, a list [xmin, ymin, xmax, ymax]

</details>

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/face_det/face_detection_res.png"/>


The explanations for the methods, parameters, etc., are as follows:

* `create_model` instantiates a face detection model (here, `PicoDet_LCNet_x2_5_face` is used as an example), and the specific explanations are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>Size of the input image; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>int/list</code></td>
<td>
<ul>
<li><b>int</b>, e.g., 640, indicating that the input image will be resized to 640x640</li>
<li><b>List</b>, e.g., [640, 512], indicating that the input image will be resized to a width of 640 and a height of 512</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering out low-confidence prediction results; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>float</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the face detection model is called for inference prediction. The `predict()` method has parameters `input`, `batch_size`, and `threshold`, which are explained as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL Link</b>, such as the web URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
<li><b>Local Directory</b>, the directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer</td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering out low-confidence prediction results; if not specified, the <code>threshold</code> parameter specified in <code>create_model</code> will be used. If <code>create_model</code> also does not specify it, the default configuration of the PaddleX official model will be used</td>
<td><code>float</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The prediction results are processed, and the prediction result for each sample is of type `dict`. It supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX's single-model inference API, please refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better face detection models. Before using PaddleX to develop face detection models, ensure you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/widerface_coco_examples.tar -P ./dataset
tar -xf ./dataset/widerface_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
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
  "dataset_path": "./dataset/example_data/widerface_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
</code></pre>
<p>The verification results mentioned above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Details of other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 1;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 500;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 100;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/face_det/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Face detection does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with <code>val_percent</code> is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of PicoDet_LCNet_x2_5_face as an example:

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`，When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves model weight files, defaulting to <code>output</code>. To specify a save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PicoDet_LCNet_x2_5_face.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction. In PaddleX, model inference prediction can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="face_detection.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. <b>Pipeline Integration</b>

The face detection module can be integrated into PaddleX pipelines such as [<b>Face Recognition</b>](../../../pipeline_usage/tutorials/cv_pipelines/face_recognition.en.md). Simply replace the model path to update the face detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the face detection module. You can refer to the Python example code in [Quick Integration](#quick), simply replace the model with the path to your trained model.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).