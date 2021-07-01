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
#include <memory>
#include <vector>

#include "encryption/include/model_code.h"
#include "encryption/include/paddle_model_encrypt.h"
#include "encryption/constant/constant_model.h"
#include "encryption/util/include/crypto/aes_gcm.h"
#include "encryption/util/include/crypto/sha256_utils.h"
#include "encryption/util/include/crypto/base64.h"
#include "encryption/util/include/system_utils.h"
#include "encryption/util/include/io_utils.h"
#include "encryption/util/include/log.h"

std::string paddle_generate_random_key() {
    std::string tmp = util::SystemUtils::random_key_iv(AES_GCM_KEY_LENGTH);
    // return util::crypto::Base64Utils::encode(tmp);
    return baidu::base::base64::base64_encode(tmp);
}

int paddle_encrypt_dir(const char* keydata,
                       const char* src_dir,
                       const char* dst_dir) {
    std::vector<std::string> files;
    int ret_files = ioutil::read_dir_files(src_dir, files);
    if (ret_files == -1) {
        return CODE_NOT_EXIST_DIR;
    }
    if (ret_files == 0) {
        return CODE_FILES_EMPTY_WITH_DIR;
    }

    // check model.yml, __model__, __params__ exist or not
    if (util::SystemUtils::check_pattern_exist(files, "model.yml")) {
        return CODE_MODEL_YML_FILE_NOT_EXIST;
    }
    if (util::SystemUtils::check_pattern_exist(files, "__model__")) {
        return CODE_MODEL_FILE_NOT_EXIST;
    }
    if (util::SystemUtils::check_pattern_exist(files, "__params__")) {
        return CODE_PARAMS_FILE_NOT_EXIST;
    }

    std::string src_str(src_dir);
    if (src_str[src_str.length() - 1] != '/') {
        src_str.append("/");
    }
    std::string dst_str(dst_dir);
    if (dst_str[dst_str.length() - 1] != '/') {
        dst_str.append("/");
    }
    int ret = CODE_OK;
    ret = ioutil::dir_exist_or_mkdir(dst_str.c_str());
    for (int i = 0; i < files.size(); ++i) {
        if (strcmp(files[i].c_str(), "__model__") == 0
            || strcmp(files[i].c_str(), "__params__") == 0
            || strcmp(files[i].c_str(), "model.yml") == 0) {
            std::string infile = src_str + files[i];
            std::string outfile = dst_str + files[i] + ".encrypted";
            ret = paddle_encrypt_model(keydata, infile.c_str(),
                                       outfile.c_str());
        } else {
            std::string infile = src_str + files[i];
            std::string outfile = dst_str + files[i];
            ret = ioutil::read_file_to_file(infile.c_str(), outfile.c_str());
        }

        if (ret != CODE_OK) {
            return ret;
        }
    }
    files.clear();
    return ret;
}

int paddle_encrypt_model(const char* keydata,
                         const char* infile,
                         const char* outfile) {
    std::string key_str =
            baidu::base::base64::base64_decode(std::string(keydata));
    if (key_str.length() != 32) {
        return CODE_KEY_LENGTH_ABNORMAL;
    }

    unsigned char* plain = NULL;
    size_t plain_len = 0;
    int ret_read = ioutil::read_file(infile, &plain, &plain_len);
    if (ret_read != CODE_OK) {
        return ret_read;
    }

    unsigned char* aes_key =
          (unsigned char*) malloc(sizeof(unsigned char) * AES_GCM_KEY_LENGTH);
    unsigned char* aes_iv =
          (unsigned char*) malloc(sizeof(unsigned char) * AES_GCM_IV_LENGTH);
    memcpy(aes_key, key_str.c_str(), AES_GCM_KEY_LENGTH);
    memcpy(aes_iv, key_str.c_str() + 16, AES_GCM_IV_LENGTH);

    unsigned char* cipher = (unsigned char*) malloc(sizeof(unsigned char) *
                                            (plain_len + AES_GCM_TAG_LENGTH));
    size_t cipher_len = 0;
    int ret_encrypt =
        util::crypto::AesGcm::encrypt_aes_gcm(plain,
                                              plain_len,
                                              aes_key,
                                              aes_iv,
                                              cipher,
                    reinterpret_cast<int&>(cipher_len));
    free(aes_key);
    free(aes_iv);
    if (ret_encrypt != CODE_OK) {
        LOGD("[M]aes encrypt ret code: %d", ret_encrypt);
        free(plain);
        free(cipher);
        return CODE_AES_GCM_ENCRYPT_FIALED;
    }

    std::string randstr = util::SystemUtils::random_str(constant::TAG_LEN);
    std::string aes_key_iv(key_str);
    std::string sha256_key_iv =
             util::crypto::SHA256Utils::sha256_string(aes_key_iv);
    for (int i = 0; i < 64; ++i) {
        randstr[i] = sha256_key_iv[i];
    }

    size_t header_len = constant::MAGIC_NUMBER_LEN +
                             constant::VERSION_LEN + constant::TAG_LEN;
    unsigned char* header =
            (unsigned char*) malloc(sizeof(unsigned char) * header_len);
    memcpy(header, constant::MAGIC_NUMBER.c_str(), constant::MAGIC_NUMBER_LEN);
    memcpy(header + constant::MAGIC_NUMBER_LEN,
           constant::VERSION.c_str(),
           constant::VERSION_LEN);
    memcpy(header + constant::MAGIC_NUMBER_LEN + constant::VERSION_LEN,
           randstr.c_str(), constant::TAG_LEN);

    int ret_write_file = ioutil::write_file(outfile, header, header_len);
    ret_write_file = ioutil::append_file(outfile, cipher, cipher_len);
    free(header);
    free(cipher);

    return ret_write_file;
}

int encrypt_stream(const std::string &keydata,
                   std::istream &in_stream, std::ostream &out_stream) {
    std::string key_str = baidu::base::base64::base64_decode(keydata);
    if (key_str.length() != 32) {
        return CODE_KEY_LENGTH_ABNORMAL;
    }

    in_stream.seekg(0, std::ios::beg);
    in_stream.seekg(0, std::ios::end);
    size_t plain_len = in_stream.tellg();
    in_stream.seekg(0, std::ios::beg);

    std::unique_ptr<unsigned char[]> plain(new unsigned char[plain_len]);
    in_stream.read(reinterpret_cast<char *>(plain.get()), plain_len);

    std::string aes_key = key_str.substr(0, AES_GCM_KEY_LENGTH);
    std::string aes_iv = key_str.substr(16, AES_GCM_IV_LENGTH);

    std::unique_ptr<unsigned char[]> cipher(
            new unsigned char[plain_len + AES_GCM_TAG_LENGTH]);
    size_t cipher_len = 0;
    int ret_encrypt = util::crypto::AesGcm::encrypt_aes_gcm(
                    plain.get(),
                    plain_len,
                    reinterpret_cast<const unsigned char*>(aes_key.c_str()),
                    reinterpret_cast<const unsigned char*>(aes_iv.c_str()),
                    cipher.get(),
                    reinterpret_cast<int&>(cipher_len));
    if (ret_encrypt != CODE_OK) {
        LOGD("[M]aes encrypt ret code: %d", ret_encrypt);
        return CODE_AES_GCM_ENCRYPT_FIALED;
    }

    std::string randstr = util::SystemUtils::random_str(constant::TAG_LEN);
    std::string aes_key_iv(key_str);
    std::string sha256_key_iv =
             util::crypto::SHA256Utils::sha256_string(aes_key_iv);
    for (int i = 0; i < 64; ++i) {
        randstr[i] = sha256_key_iv[i];
    }

    size_t header_len = constant::MAGIC_NUMBER_LEN +
                             constant::VERSION_LEN + constant::TAG_LEN;
    out_stream.write(constant::MAGIC_NUMBER.c_str(),
                     constant::MAGIC_NUMBER_LEN);
    out_stream.write(constant::VERSION.c_str(), constant::VERSION_LEN);
    out_stream.write(randstr.c_str(), constant::TAG_LEN);
    out_stream.write(reinterpret_cast<char *>(cipher.get()), cipher_len);

    return CODE_OK;
}
