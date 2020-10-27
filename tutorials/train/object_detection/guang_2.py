# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import json

from paddlex.det import transforms
import paddlex as pdx

# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=800, max_size=1333), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
#train_dataset = pdx.datasets.VOCDetection(
#    data_dir='dataset',
#    file_list='dataset/train_list.txt',
#    label_list='dataset/labels.txt',
#    transforms=train_transforms,
#    num_workers=2,
#    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    num_workers=2,
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
#num_classes = len(train_dataset.labels) + 1
#
## API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn
#model = pdx.det.FasterRCNN(num_classes=num_classes, backbone='ResNet50_vd')
#
## API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
## 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
#model.train(
#    num_epochs=36,
#    train_dataset=train_dataset,
#    train_batch_size=8,
#    eval_dataset=eval_dataset,
#    learning_rate=0.01,
#    lr_decay_epochs=[24, 33],
#    warmup_steps=1000,
#    pretrain_weights='ResNet50_vd_ssld_pretrained',
#    save_dir='output/guan_2',
#    use_vdl=False)


#eval_dataset = pdx.datasets.CocoDetection(
#    data_dir='dataset_coco/JPEGImages',
#    ann_file='dataset_coco/val.json',
#    num_workers=2,
#    transforms=eval_transforms)
#model = pdx.load_model('output/guan_4/best_model/')
#eval_details = model.evaluate(eval_dataset, batch_size=8, return_details=True)
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


with open('output/guan_4/best_model/eval_details.json', 'r') as f:
    eval_details = json.load(f)
json_path = 'output/guan_4/best_model/gt.json'
json.dump(eval_details['gt'], open(json_path, "w"), indent=4, cls=MyEncoder)
json_path = 'output/guan_4/best_model/bbox.json'
json.dump(eval_details['bbox'], open(json_path, "w"), indent=4, cls=MyEncoder)
