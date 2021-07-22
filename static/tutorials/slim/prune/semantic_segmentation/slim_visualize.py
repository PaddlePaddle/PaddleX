import paddlex as pdx
model = pdx.load_model('output/unet/best_model')
pdx.slim.visualize(model, 'unet.sensi.data', save_dir='./')
