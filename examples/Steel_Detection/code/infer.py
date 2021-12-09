import paddlex as pdx
import os
import cv2

pic_index = 25

# 获取测试文件名
test_path = 'steel/test_list.txt'
f = open(test_path)
lines = f.readlines()
imgname = os.path.basename(lines[pic_index].split(' ')[0])
labelname = os.path.basename(lines[pic_index].split(' ')[1])
labelname = labelname.replace('\n', '')
f.close()

# 加载模型并预测
image_path = os.path.join('steel/JPEGImages', imgname)
model = pdx.load_model('output/hrnet/best_model')
result = model.predict(image_path)
pdx.seg.visualize(image_path, result, weight=0.4, save_dir='output/predict')

# 输出ground truth
label_path = os.path.join('steel/Annotations', labelname)
mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
gt = {'label_map': mask}
pdx.seg.visualize(image_path, gt, weight=0.4, save_dir='output/gt')
