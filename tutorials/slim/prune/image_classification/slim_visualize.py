import paddlex as pdx
model = pdx.load_model('output/mobilenetv2/best_model')
pdx.slim.visualize(model, 'mobilenetv2.sensi.data', save_dir='./')
