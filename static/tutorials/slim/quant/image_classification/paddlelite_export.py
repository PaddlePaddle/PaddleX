# 需先安装paddlelite
import paddlelite.lite as lite

model_filename = 'server_mobilenet/__model__'
params_filename = 'server_mobilenet/__params__'
export_filename = 'mobilenetv2'

opt = lite.Opt()
# 将正常模型导出为Lite模型
opt.run_optimize("", model_filename, params_filename, 'naive_buffer', 'arm',
                 export_filename)

quant_model_filename = 'quant_mobilenet/__model__'
quant_params_filename = 'quant_mobilenet/__params__'
quant_export_filename = 'mobilenetv2_quant'

# 将量化模型导出为Lite模型
opt.run_optimize("", quant_model_filename, quant_params_filename,
                 'naive_buffer', 'arm', quant_export_filename)
