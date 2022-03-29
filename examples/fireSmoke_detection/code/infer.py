# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
predictor = pdx.deploy.Predictor(model_dir='frcnn_dcn_inference_model/inference_model',
                                    use_gpu=True)
result = predictor.predict(img_file='test_imgs/fire_1.jpg',
                           warmup_iters=100,
                           repeats=200)
