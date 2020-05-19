#include <stdio.h>
#include "paddle/paddle_inference_api.h"

#ifndef PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H
#define PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * load (un)encrypted model and params to paddle::AnalysisConfig
 * @param config
 * @param key           加解密key（注：该SDK能符合的key信息为32字节转为BASE64编码后才能通过）
 * @param model_file    模型文件路径
 * @param param_file    参数文件路径
 * @return              error_code
 */
int paddle_security_load_model(paddle::AnalysisConfig* config,
                               const char* key,
                               const char* model_file,
                               const char* param_file);
/**
 * check file (un)encrypted?
 * @param file_path
 * @return
 */
int paddle_check_file_encrypted(const char* file_path);

#ifdef __cplusplus
}
#endif

#endif //PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H
