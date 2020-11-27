import paddlex as pdx
model = pdx.load_model('output/yolov3_mobilenetv1/best_model')
pdx.slim.visualize(model, 'yolov3.sensi.data', save_dir='./')
