import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

model = pdx.load_model('output/yolov3_mobilenetv1/best_model')

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/val_list.txt',
    label_list='insect_det/labels.txt',
    transforms=model.eval_transforms)

pdx.slim.prune.analysis(
    model, dataset=eval_dataset, batch_size=8, save_file='yolov3.sensi.data')
