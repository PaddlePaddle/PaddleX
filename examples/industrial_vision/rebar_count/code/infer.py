import glob
import numpy as np
import threading
import time
import random
import os
import base64
import cv2
import json
import paddlex as pdx

# 待预测图片路径
image_name = 'work/dataset_reinforcing_steel_bar_counting/JPEGImages/4B145787.jpg'

# 预测模型加载
model = pdx.load_model('output/yolov3_mobilnetv1/best_model')

# 读取图片与获取预测结果
img = cv2.imread(image_name)
result = model.predict(img)

# 解析预测结果，并保存到txt中
keep_results = []
areas = []
f = open('result.txt', 'a')
count = 0
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:
        continue
    keep_results.append(dt)
    count += 1
    f.write(str(dt) + '\n')
    f.write('\n')
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)
print(count)
f.write("the total number is :" + str(int(count)))
f.close()

# 可视化保存
pdx.det.visualize(
    image_name, result, threshold=0.5, save_dir='./output/yolov3_mobilnetv1')