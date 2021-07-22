import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

model = pdx.load_model('output/mobilenetv2/best_model')

eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=model.eval_transforms)

pdx.slim.prune.analysis(
    model,
    dataset=eval_dataset,
    batch_size=16,
    save_file='mobilenetv2.sensi.data')
