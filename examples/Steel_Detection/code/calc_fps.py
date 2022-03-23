import paddlex as pdx
import time
import glob

image_names = glob.glob('steel/JPEGImages/*.jpg')
model = pdx.load_model('output/hrnet/best_model')
start_time = 0
for i, image_name in enumerate(image_names):
    if i == 100:  # 前面100张不计时，用于warm up
        start_time = time.time()
    if i > 299:
        break
    result = model.predict(image_name)
    print(i)
fps = (time.time() - start_time) / 200
fps = 1 / fps
print(f"fps:{fps}")
