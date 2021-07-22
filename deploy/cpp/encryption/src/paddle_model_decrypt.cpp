//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <string.h>

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <iterator>
#include <algorithm>

#include "encryption/include/paddle_model_decrypt.h"
#include "encryption/include/model_code.h"
#include "encryption/util/include/crypto/aes_gcm.h"
#include "encryption/util/include/crypto/base64.h"
#include "encryption/util/include/io_utils.h"
#include "encryption/util/include/log.h"
#include "encryption/util/include/constant/constant_model.h"
#include "encryption/util/include/system_utils.h"

/**
 * 0 - encrypted
 * 1 - unencrypt
 */
int paddle_check_file_encrypted(const char* file_path) {
  return util::SystemUtils::check_file_encrypted(file_path);
}

std::string decrypt_file(const char* file_path, const char* key) {
  int ret = paddle_check_file_encrypted(file_path);
  if (ret != CODE_OK) {
    LOGD("[M]check file encrypted failed, code: %d", ret);
    return std::string();
  }

// std::string key_str = util::crypto::Base64Utils::decode(std::string(key));
  std::string key_str = baidu::base::base64::base64_decode(std::string(key));
  int ret_check =
          util::SystemUtils::check_key_match(key_str.c_str(), file_path);
  if (ret_check != CODE_OK) {
    LOGD("[M]check key failed in decrypt_file, code: %d", ret_check);
    return std::string();
  }
  unsigned char* aes_key =
          (unsigned char*) malloc(sizeof(unsigned char) * AES_GCM_KEY_LENGTH);
  unsigned char* aes_iv =
          (unsigned char*) malloc(sizeof(unsigned char) * AES_GCM_IV_LENGTH);
  memcpy(aes_key, key_str.c_str(), AES_GCM_KEY_LENGTH);
  memcpy(aes_iv, key_str.c_str() + 16, AES_GCM_IV_LENGTH);

  size_t pos = constant::MAGIC_NUMBER_LEN +
                    constant::VERSION_LEN + constant::TAG_LEN;

  // read encrypted data
  unsigned char* dataptr = NULL;
  size_t data_len = 0;
  int ret_read_data = ioutil::read_with_pos(file_path, pos,
                                            &dataptr, &data_len);
  if (ret_read_data != CODE_OK) {
    LOGD("[M]read file failed, code = %d", ret_read_data);
    return std::string();
  }

  // decrypt model data
  size_t model_plain_len = data_len - AES_GCM_TAG_LENGTH;
  unsigned char* model_plain =
            (unsigned char*) malloc(sizeof(unsigned char) * model_plain_len);

  int ret_decrypt_file =
        util::crypto::AesGcm::decrypt_aes_gcm(
            dataptr,
            data_len,
            aes_key,
            aes_iv,
            model_plain,
            reinterpret_cast<int&>(model_plain_len));
  free(dataptr);
  free(aes_key);
  free(aes_iv);
  if (ret_decrypt_file != CODE_OK) {
    free(model_plain);
    LOGD("[M]decrypt file failed, decrypt ret = %d", ret_decrypt_file);
    return std::string();
  }
  std::string result((const char*)model_plain, model_plain_len);
  free(model_plain);
  return result;
}

/**
 * support model_file encrypted or unencrypt
 * support params_file encrypted or unencrypt
 * all in one interface
 */
/*
int paddle_security_load_model(
    paddle::AnalysisConfig* config,
    const char* key,
    const char* model_file,
    const char* param_file) {

    // 0 - file encrypted   1 - file unencrypted
    int m_en_flag = util::SystemUtils::check_file_encrypted(model_file);
    if (m_en_flag == CODE_OPEN_FAILED) {
        return m_en_flag;
    }
    int p_en_flag = util::SystemUtils::check_file_encrypted(param_file);
    if (p_en_flag == CODE_OPEN_FAILED) {
        return p_en_flag;
    }

    unsigned char* aes_key = NULL;
    unsigned char* aes_iv = NULL;
    if (m_en_flag == 0 || p_en_flag == 0) {
//  std::string key_str = util::crypto::Base64Utils::decode(std::string(key));
        std::string key_str =
                baidu::base::base64::base64_decode(std::string(key));
        int ret_check = 0;
        if (m_en_flag == 0) {
            ret_check = util::SystemUtils::check_key_match(key_str.c_str(),
                                                           model_file);
            if (ret_check != CODE_OK) {
                LOGD("[M]check key failed in model_file");
                return ret_check;
            }
        }

        if (p_en_flag == 0) {
            ret_check = util::SystemUtils::check_key_match(key_str.c_str(),
                                                           param_file);
            if (ret_check != CODE_OK) {
                LOGD("[M]check key failed in param_file");
                return ret_check;
            }
        }
        aes_key =
           (unsigned char*) malloc(sizeof(unsigned char) * AES_GCM_KEY_LENGTH);
        aes_iv =
           (unsigned char*) malloc(sizeof(unsigned char) * AES_GCM_IV_LENGTH);
        memcpy(aes_key, key_str.c_str(), AES_GCM_KEY_LENGTH);
        memcpy(aes_iv, key_str.c_str() + 16, AES_GCM_IV_LENGTH);
    }

    size_t pos = constant::MAGIC_NUMBER_LEN +
                      constant::VERSION_LEN + constant::TAG_LEN;

    // read encrypted model
    unsigned char* model_dataptr = NULL;
    size_t model_data_len = 0;
    int ret_read_model = ioutil::read_with_pos(model_file, pos,
                                       &model_dataptr, &model_data_len);
    if (ret_read_model != CODE_OK) {
        LOGD("[M]read model failed");
        return ret_read_model;
    }

    size_t model_plain_len = 0;
    unsigned char* model_plain = NULL;
    if (m_en_flag == 0) {
        // decrypt model data
        model_plain_len = model_data_len - AES_GCM_TAG_LENGTH;
        model_plain =
            (unsigned char*) malloc(sizeof(unsigned char) * model_plain_len);

        int ret_decrypt_model =
            util::crypto::AesGcm::decrypt_aes_gcm(model_dataptr,
                                                  model_data_len,
                                                  aes_key,
                                                  aes_iv,
                                                  model_plain,
                      reinterpret_cast<int&>(model_plain_len));
        free(model_dataptr);
        if (ret_decrypt_model != CODE_OK) {
            free(aes_key);
            free(aes_iv);
            free(model_plain);
            LOGD("[M]decrypt model failed, decrypt ret = %d",
                                           ret_decrypt_model);
            return CODE_AES_GCM_DECRYPT_FIALED;
        }
    } else {
        model_plain = model_dataptr;
        model_plain_len = model_data_len;
    }


    // read encrypted params
    unsigned char* params_dataptr = NULL;
    size_t params_data_len = 0;
    int ret_read_params = ioutil::read_with_pos(param_file, pos,
                                                &params_dataptr,
                                                &params_data_len);
    if (ret_read_params != CODE_OK) {
        LOGD("[M]read params failed");
        return ret_read_params;
    }

    size_t params_plain_len = 0;
    unsigned char* params_plain = NULL;
    if (p_en_flag == 0) {
        // decrypt params data
        params_plain_len = params_data_len - AES_GCM_TAG_LENGTH;
        params_plain =
            (unsigned char*) malloc(sizeof(unsigned char) * params_plain_len);

        int ret_decrypt_params =
            util::crypto::AesGcm::decrypt_aes_gcm(params_dataptr,
                                                  params_data_len,
                                                  aes_key,
                                                  aes_iv,
                                                  params_plain,
                      reinterpret_cast<int&>(params_plain_len));
        free(params_dataptr);
        free(aes_key);
        free(aes_iv);
        if (ret_decrypt_params != CODE_OK) {
            free(params_plain);
            LOGD("[M]decrypt params failed, decrypt ret = %d",
                  ret_decrypt_params);
            return CODE_AES_GCM_DECRYPT_FIALED;
        }
    } else {
        params_plain = params_dataptr;
        params_plain_len = params_data_len;
    }

    LOGD("Prepare to set config");

    config->SetModelBuffer(reinterpret_cast<const char*>(model_plain),
                           model_plain_len,
                           reinterpret_cast<const char*>(params_plain),
                           params_plain_len);

    if (m_en_flag == 1) {
        free(model_dataptr);
    }

    if (p_en_flag == 1) {
        free(params_dataptr);
    }

    return CODE_OK;
}
*/
