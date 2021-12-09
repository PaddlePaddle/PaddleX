import paddlex
import cv2
import numpy as np

from point_tools import parse_mask_edge_points, visualize_mask_edge

model = paddlex.load_model('output/mask_rcnn_r50_fpn/best_model')

img = cv2.imread('dataset/JPEGImages/Image_20210615204210757.bmp')
result = model.predict(
    'dataset/JPEGImages/Image_20210615204210757.bmp',
    transforms=model.test_transforms)

mask_edge_points = parse_mask_edge_points(result, score_threshold=0.95)
img = visualize_mask_edge(
    img, mask_edge_points=mask_edge_points, point_size=1, color=(0, 0, 255))
cv2.imwrite('./test.png', img)
