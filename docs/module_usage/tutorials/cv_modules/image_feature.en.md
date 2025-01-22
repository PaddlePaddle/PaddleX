---
comments: true
---

# Image Feature Module Development Tutorial

## I. Overview
The image feature module is one of the important tasks in computer vision, primarily referring to the automatic extraction of useful features from image data using deep learning methods, to facilitate subsequent image retrieval tasks. The performance of this module directly affects the accuracy and efficiency of the subsequent tasks. In practical applications, image features typically output a set of feature vectors, which can effectively represent the content, structure, texture, and other information of the image, and will be passed as input to the subsequent retrieval module for processing.

## II. Supported Model List


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recall@1 (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-ShiTuV2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">Trained Model</a></td>
<td>84.2</td>
<td>5.23428</td>
<td>19.6005</td>
<td>16.3 M</td>
<td rowspan="3">PP-ShiTuV2 is a general image feature system consisting of three modules: object detection, feature extraction, and vector retrieval. These models are part of the feature extraction module and can be selected based on system requirements.</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">Trained Model</a></td>
<td>88.69</td>
<td>13.1957</td>
<td>285.493</td>
<td>306.6 M</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">Trained Model</a></td>
<td>91.03</td>
<td>51.1284</td>
<td>1131.28</td>
<td>1.05 G</td>
</tr>
</table>

<b>Note: The above accuracy metrics are Recall@1 from AliProducts. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the image feature module. You can switch between models under this module freely, and you can also integrate the model inference of the image feature module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "PP-ShiTuV2_rec"
model = create_model(model_name)
output = model.predict("general_image_recognition_001.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_json("./output/res.json")
```

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'general_image_recognition_001.jpg', 'feature': [0.049109019339084625, 0.003743218956515193, 0.005219039972871542, -0.02721826359629631, -0.016909820958971977, -0.006206876132637262, -0.04125503450632095, -0.0019930789712816477, -0.022831404581665993, -0.047313764691352844, 0.060403138399124146, 0.0761566013097763, 0.0017959520919248462, 0.0493767075240612, -0.021170074120163918, -0.03655105456709862, 0.02802395075559616, 0.039860036224126816, -0.02787216380238533, 0.02957169897854328, 0.06538619846105576, 0.006429907865822315, 0.007104719523340464, -0.005089965183287859, -0.05652903392910957, -0.005592004396021366, 0.03234156593680382, 0.07858140766620636, 0.031976014375686646, 0.009858119301497936, -0.005774488672614098, -0.06363064795732498, 0.022391995415091515, 0.01687457598745823, -0.023193085566163063, -0.04410340636968613, -0.048582229763269424, -0.005215164739638567, -0.045220136642456055, 0.02273370884358883, 0.1008470356464386, 0.020996153354644775, 0.04300517961382866, 0.012766947969794273, 0.0331314392387867, -0.025111157447099686, 0.04866917431354523, -0.0007637099479325116, 0.09444739669561386, -0.11289195716381073, -0.07322289794683456, -0.0777730941772461, -0.015321447513997555, -0.004911054391413927, 0.007349034305661917, -0.06889548897743225, -0.011488647200167179, 0.0266891997307539, 0.015688180923461914, -0.06728753447532654, 0.03612099587917328, 0.025985529646277428, 0.03768249973654747, -0.08363037556409836, -0.048576220870018005, 0.08420056849718094, 0.023980043828487396, 0.07883623242378235, -0.013208202086389065, -0.018564118072390556, -0.017457934096455574, -0.020058095455169678, -0.02541436441242695, -0.008189533837139606, -0.056144408881664276, -0.02954513393342495, 0.010910517536103725, 0.003301263554021716, 0.06621509790420532, -0.06254079937934875, 0.06691820919513702, -0.061315152794122696, -0.0664873942732811, 0.03236202895641327, 0.034841813147068024, -0.013619803823530674, 0.10573607683181763, 0.02072797901928425, 0.055478695780038834, 0.010184620507061481, -0.05034280940890312, -0.02191540040075779, 0.047144751995801926, -0.031101807951927185, -0.016248639672994614, 0.05873332917690277, 0.015815891325473785, -0.001074428902938962, -0.010065405629575253, -0.045008182525634766, 0.03385455906391144, 0.0015282221138477325, -0.033003728836774826, -0.03174556791782379, 0.11101125925779343, -0.006336087826639414, 0.01115291379392147, 0.05289424583315849, -0.031148016452789307, -0.025501884520053864, -0.04285573586821556, -0.025347966700792313, 0.07360142469406128, -0.021775048226118088, 0.06776975840330124, -0.027090832591056824, 0.004754350520670414, 0.020962830632925034, -0.015184056013822556, 0.0010469526750966907, 0.10906203091144562, 0.023142946884036064, -0.014811648055911064, 0.060142189264297485, 0.00820994470268488, -0.02131684496998787, 0.0638369619846344, -0.04250387102365494, 0.008871919475495815, -0.008007975295186043, 0.07150974124670029, 0.011682109907269478, -0.006690092850476503, 0.011732078157365322, -0.07107444852590561, -0.06642609089612961, -0.011343806982040405, -0.006821080576628447, -0.02998213842511177, 0.028652319684624672, 0.03125355765223503, 0.07194835692644119, 0.07021243870258331, 0.002937359269708395, -0.0499102883040905, 0.040573444217443466, -0.019794275984168053, 0.03276952728629112, -0.11937176436185837, -0.05637907609343529, -0.016590707004070282, -0.020010240375995636, 0.026032714173197746, 0.0035290969535708427, -0.02005072496831417, 0.010683903470635414, 0.016347797587513924, 0.015019580721855164, -0.013746236450970173, 0.059569939970970154, -0.049140941351652145, -0.0019614780321717262, -0.08775275200605392, -0.10722333192825317, -0.042984429746866226, -0.00020981401030439883, 0.019572105258703232, -0.06898674368858337, 0.01378229632973671, 0.010090887546539307, 0.004062692169100046, 0.03605732321739197, -0.030028242617845535, -0.071064792573452, 2.3956217773957178e-05, -0.04649794474244118, 0.006212098523974419, 0.05053247511386871, 0.017688272520899773, 0.06759831309318542, 0.06286999583244324, -0.0010359658626839519, 0.02886095643043518, 0.07879934459924698, -0.04438954219222069, 0.03000231273472309, 0.0032854368910193443, 0.04237217828631401, -0.019776295870542526, -0.003197435289621353, -0.029772961512207985, 0.014659246429800987, 0.007493179757148027, -0.05654020234942436, -0.06438001990318298, 0.09076389670372009, 0.06214659661054611, 0.004840471316128969, 0.045114848762750626, 0.07397229224443436, -0.032566577196121216, -0.02464713528752327, -0.001989303156733513, 0.011997431516647339, -0.05213696509599686, -0.016684064641594887, -0.025001073256134987, -0.06234518438577652, 0.08302164077758789, 0.06388438493013382, -0.02603762038052082, -0.057507626712322235, 0.010737594217061996, 0.021288368850946426, 0.050199754536151886, 0.020688340067863464, -0.03297201544046402, 0.046142708510160446, -0.010062780231237411, -0.009058497846126556, -0.028288882225751877, 0.04905378818511963, 0.014915363863110542, 0.013268127106130123, 0.0682050809264183, -0.05951741710305214, 0.057072725147008896, -0.05045686662197113, -0.06781881302595139, 0.013548677787184715, 0.05480438843369484, 0.004949226509779692, -0.06020767241716385, 0.06817059963941574, -0.03284472972154617, 0.014837299473583698, -0.02967672049999237, 0.01816580630838871, 0.02033187262713909, 0.02823079377412796, -0.04500351473689079, 0.08674795180559158, -0.029325610026717186, 0.023551611229777336, 0.06388993561267853, -0.009744416922330856, -0.03457322716712952, 0.04782603681087494, 0.02554101124405861, -0.0456324964761734, -0.028274627402424812, 0.06662072241306305, -0.05609177425503731, 0.05615284666419029, 0.019022393971681595, -0.0025437776930630207, 0.009997396729886532, -0.02311607636511326, -0.01859533227980137, 0.014079087413847446, -0.004390522837638855, 0.033753011375665665, -0.01057991199195385, 0.06405404955148697, -0.044001828879117966, 0.030532050877809525, 0.07704707980155945, -0.008154668845236301, 0.07150844484567642, -0.025875508785247803, 0.012938959524035454, 0.031110644340515137, 0.040104109793901443, -0.015837285667657852, 0.013013459742069244, -0.04932061582803726, 0.1163453534245491, -0.018216004595160484, 0.09031085669994354, -0.00798568595200777, 0.03739010915160179, 0.009020711295306683, 0.0121764512732625, 0.05221538990736008, -0.039455823600292206, 0.0006922244210727513, 0.015570797957479954, 0.07312478870153427, -0.020688941702246666, -0.013934718444943428, 0.008533086627721786, 0.06440478563308716, -0.01079096831381321, 0.01836889237165451, 0.02253551408648491, 0.013003944419324398, -0.09377135336399078, -0.0679609626531601, 0.08662433922290802, 0.04313892871141434, -0.0991244912147522, 0.031046167016029358, -0.00529451621696353, 0.03583913296461105, -0.036976899951696396, -0.02047165483236313, 0.05907558649778366, -0.01609981805086136, 0.025996552780270576, -0.020804448053240776, -0.08788128197193146, 0.008795336820185184, 0.02871711179614067, 0.03529304638504982, 0.03294558450579643, -0.042029064148664474, 0.06038316339254379, -0.04894087836146355, 0.062449462711811066, 0.006345283705741167, -0.05110999941825867, -0.05251497030258179, -0.0485706627368927, 0.01973171904683113, 0.008282365277409554, -0.07566660642623901, -0.054833680391311646, -0.07128996402025223, 0.04532742500305176, 0.03617053106427193, -0.08390213549137115, 0.059522684663534164, -0.0420474074780941, -0.01778397522866726, -0.0009484086185693741, -0.0005348647246137261, 0.060148268938064575, -0.0330856516957283, -0.0246503297239542, 0.017774641513824463, 0.030457578599452972, 0.05355843901634216, -0.0587775893509388, -0.10057137906551361, 0.018445275723934174, -0.009815521538257599, -0.018039006739854813, 0.07852229475975037, 0.06736239790916443, -0.001333046704530716, 0.007442455273121595, 0.024762822315096855, 0.01432487741112709, -0.015761420130729675, -0.06996625661849976, -0.046995896846055984, 0.007052265573292971, 0.01292750146239996, -0.040947142988443375, -0.04076245054602623, 0.02297302521765232, 0.031471725553274155, -0.00020813643641304225, 0.038925208151340485, -0.06558586657047272, -0.007383601740002632, -0.023945683613419533, -0.00975192990154028, -0.02227136865258217, 0.04719878360629082, -0.06225449591875076, -0.01973516121506691, -0.03993571922183037, 0.05859784781932831, -0.033084310591220856, -0.09690506756305695, -0.023898473009467125, -0.021453123539686203, -0.03663061931729317, -0.017285870388150215, 0.03141544386744499, 0.03807565197348595, -0.026278872042894363, -0.01809430494904518, -0.015841489657759666, 0.012133757583796978, 0.015615759417414665, 0.025031639263033867, 0.024692615494132042, 0.04616452008485794, -0.060463856905698776, -0.015853295102715492, -0.018289266154170036, -0.04386031627655029, 0.03577912971377373, -0.027260515838861465, 0.0262057613581419, 0.057801421731710434, -0.03247777372598648, -0.025150176137685776, 0.00601597735658288, 0.04657217860221863, -0.080239437520504, 0.03466523066163063, -0.030453147366642952, 0.0482921376824379, -0.08164504170417786, -0.014207101427018642, 0.009327770210802555, -0.04090266674757004, 0.030108662322163582, 0.009122633375227451, -0.014895373024046421, -0.05183123052120209, -0.013794025406241417, 0.07302702963352203, -0.09872224181890488, -0.018709152936935425, 0.0033975779078900814, -0.014089024625718594, 0.03661659359931946, 0.057056576013565063, 0.020562296733260155, 0.043459419161081314, 0.01512663159519434, -0.08066604286432266, -0.004175281152129173, -0.01924264058470726, 0.019628198817372322, 0.0038949784357100725, 0.0417584627866745, 0.04380005970597267, 0.07349026203155518, 0.014636299572885036, -0.05216812714934349, -0.012727170251309872, 0.045851919800043106, -0.01108911819756031, 0.03997885435819626, -0.030407968908548355, -0.010511808097362518, -0.05589114874601364, 0.044310227036476135, 0.07810009270906448, -0.04502151906490326, -0.04358488321304321, -0.05491747707128525, -0.03352531045675278, 0.05367979034781456, -0.051684021949768066, -0.09159758687019348, -0.06795314699411392, 0.05151400715112686, -0.03667590394616127, 0.0415973998606205, 0.004540375899523497, 0.05324205011129379, 0.007202384993433952, -0.04345241189002991, 0.050976917147636414, -0.04033355042338371, 0.06330423802137375, 0.05229709669947624, 0.07360764592885971, -0.032564569264650345, 0.05199885368347168, -0.00532008009031415, -0.0022711465135216713, 0.02948114089667797, -0.016920195892453194, -0.02260630950331688, 0.028158463537693024, 0.07792772352695465, -0.018052909523248672, -0.03556442633271217, 0.00789704080671072, -0.07626558840274811, -0.03160900995135307, 0.01614975929260254, 0.029233204200863838, -0.019288279116153717, 0.042064376175403595, 0.02296893298625946, -0.02441602759063244, 0.0140019990503788, -0.06483341753482819, 0.11968966573476791, 0.03714313730597496, -0.06824050098657608, -0.016793427988886833, 0.009564070962369442, -0.09704745560884476, 0.01676984317600727, -0.02669912949204445, 0.0003774994402192533, -0.06364470720291138, 0.01929471455514431, -0.009203671477735043, -0.003950135316699743, -0.03651873394846916, -0.013927181251347065, -0.0008955051307566464, 0.02604137919843197, -0.07753892242908478, 0.015690961852669716, 0.0001985478593269363, -0.029516037553548813, -0.028976714238524437, -0.07917939871549606, 0.036289919167757034, 0.03013240359723568, -0.07124550640583038]}}
```

The meanings of the parameters are as follows:
- `input_path`: The path to the input image to be predicted.
- `feature`: A list of floating-point numbers representing the feature vector, with a length of 512.

</details>

Descriptions of related methods, parameters, etc., are as follows:

* The `create_model` method instantiates an image feature model (using `PP-ShiTuV2_rec` as an example). Specific descriptions are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>The name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. When `model_name` is specified, PaddleX's built-in model parameters are used by default. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the image feature model is called for inference. The parameters of the `predict()` method are `input` and `batch_size`, with specific descriptions as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
  <li><b>Local directory</b>, which must contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>Dictionary</b>, where the <code>key</code> must correspond to the specific task (e.g., <code>"img"</code> for image classification), and the <code>value</code> supports the above types of data, for example: <code>{"img": "/root/data1"}</code></li>
  <li><b>List</b>, elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code></li>
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
</table>

* Process the prediction results. Each sample's prediction result is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Explanation</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a <code>json</code> file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is provided, the saved file name matches the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* Additionally, the prediction results can be accessed via properties, as follows:

<table>
<thead>
<tr>
<th>Property</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td><code>json</code></td>
<td>Get the prediction result in <code>json</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better image feature models. Before developing image feature models with PaddleX, ensure you have installed the classification-related model training plugins for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>.  Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Image Feature Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/image_feature.en.md).


#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:
```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Inshop_examples.tar -P ./dataset
tar -xf ./dataset/Inshop_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
A single command can complete data validation:
```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 1000,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/05_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/02_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/02_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/04_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/04_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/12_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/07_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/04_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/04_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/01_1_front.jpg&quot;
    ],
    &quot;gallery_samples&quot;: 110,
    &quot;gallery_sample_paths&quot;: [
      &quot;check_dataset/demo_img/06_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/01_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/04_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/02_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/02_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/02_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/02_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/03_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/02_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/03_2_side.jpg&quot;
    ],
    &quot;query_samples&quot;: 125,
    &quot;query_sample_paths&quot;: [
      &quot;check_dataset/demo_img/08_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/01_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/02_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/04_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/09_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/04_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/02_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/06_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/02_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/02_2_side.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/Inshop_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;ShiTuRecDataset&quot;
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:
* <code>attributes.train_samples</code>: The number of training samples in this dataset is 1000;
* <code>attributes.gallery_samples</code>: The number of gallery (or reference) samples in this dataset is 110;
* <code>attributes.query_samples</code>: The number of query samples in this dataset is 125;
* <code>attributes.train_sample_paths</code>: A list of relative paths to the visual images of training samples in this dataset;
* <code>attributes.gallery_sample_paths</code>: A list of relative paths to the visual images of gallery (or reference) samples in this dataset;
* <code>attributes.query_sample_paths</code>: A list of relative paths to the visual images of query samples in this dataset;</p>
<p>Additionally, the dataset verification also analyzes the number of images and image categories within the dataset, and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/img_recognition/01.png"></p></details>

### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data verification, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>The image feature task supports converting <code>LabelMe</code> format datasets to <code>ShiTuRecDataset</code> format. The parameters for dataset format conversion can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example parameter descriptions in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion. The image feature task supports converting <code>LabelMe</code> format datasets to <code>ShiTuRecDataset</code> format, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, the source dataset format needs to be set, default is <code>null</code>, optional value is <code>LabelMe</code>;</li>
</ul>
<p>For example, if you want to convert a <code>LabelMe</code> format dataset to <code>ShiTuRecDataset</code> format, you need to modify the configuration file as follows:</p>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/image_classification_labelme_examples.tar -P ./dataset
tar -xf ./dataset/image_classification_labelme_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples
</code></pre>
<p>After the data conversion is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Some example parameter descriptions in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When <code>True</code>, the dataset will be re-split, default is <code>False</code>;</li>
<li><code>train_percent</code>: If the dataset is re-split, the percentage of the training set needs to be set, the type is any integer between 0-100, and it needs to ensure that the sum of <code>gallery_percent</code> and <code>query_percent</code> values is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with 70% training set, 20% gallery set, and 10% query set, you need to modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 70
    gallery_percent: 20
    query_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=70 \
    -o CheckDataset.split.gallery_percent=20 \
    -o CheckDataset.split.query_percent=10
</code></pre>
<blockquote>
<p>❗Note: Due to the specificity of image feature model evaluation, data partitioning is meaningful only when the train, query, and gallery sets belong to the same category system. During the evaluation of recognition models, it is imperative that the gallery and query sets belong to the same category system, which may or may not be the same as the train set. If the gallery and query sets do not belong to the same category system as the train set, the evaluation after data partitioning becomes meaningless. It is recommended to proceed with caution.</p>
</blockquote></details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of the image feature model PP-ShiTuV2_rec as an example:

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`，When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves the model weight files, with the default being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
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

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file, detailed instructions can be found in [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including recall1、recall5、mAP；</p></details>

### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.


#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg) to your local machine.

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_image_recognition_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`.
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

> ❗ Note: The inference result of the recognition model is a set of vectors, which requires a retrieval module to complete image feature.

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.<b>Pipeline Integration</b>

The image feature module can be integrated into the <b>General Image Recognition Pipeline</b> (comming soon) of PaddleX. Simply replace the model path to update the image feature module of the relevant pipeline. In pipeline integration, you can use service-oriented deployment to deploy your trained model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the image feature module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.
