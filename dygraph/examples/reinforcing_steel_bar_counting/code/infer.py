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
 
image_name = 'dataset/JPEGImages/6B898244.jpg'

model_gige2= pdx.load_model('output/inference_model/inference_model')
 
 
img = cv2.imread(image_name)
results1 = model_gige2.predict(img)
 
keep_results = []
areas = []
f = open('result.txt','a')
count = 0
for dt in np.array(results1):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:
        continue
    keep_results.append(dt)
    count+=1
    f.write(str(dt)+'\n')
    f.write('\n')
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)
print(count)
f.write("the total number is :"+str(int(count)))
f.close()

import paddlex as pdx
model = pdx.load_model('output/ppyolo_r50vd_dcn/best_model')

result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./output/ppyolo_r50vd_dcn')