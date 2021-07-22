import paddlex as pdx

clip_min_value = [7172, 6561, 5777, 5103, 4291, 4000, 4000, 4232, 6934, 7199]
clip_max_value = [
    50000, 50000, 50000, 50000, 50000, 40000, 30000, 18000, 40000, 36000
]
data_info_file = 'dataset/remote_sensing_seg/train_infomation.pkl'

train_analysis = pdx.datasets.analysis.Seg(
    data_dir='dataset/remote_sensing_seg',
    file_list='dataset/remote_sensing_seg/train.txt',
    label_list='dataset/remote_sensing_seg/labels.txt')

train_analysis.cal_clipped_mean_std(clip_min_value, clip_max_value,
                                    data_info_file)
