import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

model = pdx.load_model('output/unet/best_model')

eval_dataset = pdx.datasets.SegDataset(
    data_dir='optic_disc_seg',
    file_list='optic_disc_seg/val_list.txt',
    label_list='optic_disc_seg/labels.txt',
    transforms=model.eval_transforms)

pdx.slim.prune.analysis(
    model, dataset=eval_dataset, batch_size=4, save_file='unet.sensi.data')
