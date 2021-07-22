import paddlex as pdx

train_analysis = pdx.datasets.analysis.Seg(
    data_dir='dataset/remote_sensing_seg',
    file_list='dataset/remote_sensing_seg/train.txt',
    label_list='dataset/remote_sensing_seg/labels.txt')

train_analysis.analysis()
