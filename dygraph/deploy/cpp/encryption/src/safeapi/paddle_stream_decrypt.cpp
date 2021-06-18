#include <iostream>
#include <string>
#include <string.h>
#include <fstream>
#include <memory>
#include <iterator>
#include <algorithm>
#include "paddle_stream_decrypt.h"
#include "model_code.h"
#include "../util/crypto/aes_gcm.h"
#include "../util/io_utils.h"
#include "../util/log.h"
#include "../constant/constant_model.h"
#include "../util/system_utils.h"
#include "../util/crypto/base64.h"

int paddle_check_stream_encrypted(std::istream &cipher_stream) {
    return util::SystemUtils::check_file_encrypted(cipher_stream);
}

int decrypt_stream(std::istream &cipher_stream, std::ostream &plain_stream, const std::string &key_base64){
    int ret = paddle_check_stream_encrypted(cipher_stream);
    if (ret != CODE_OK) {
        LOGD("[M]check file encrypted failed, code: %d", ret);
        return ret;
    }

    std::string key_str = baidu::base::base64::base64_decode(key_base64.c_str());
    int ret_check = util::SystemUtils::check_key_match(key_str, cipher_stream);
    if (ret_check != CODE_OK) {
        LOGD("[M]check key failed in decrypt_file, code: %d", ret_check);
        return CODE_KEY_NOT_MATCH;
    }

    std::string aes_key = key_str.substr(0, AES_GCM_KEY_LENGTH);
    std::string aes_iv = key_str.substr(16, AES_GCM_IV_LENGTH);

    cipher_stream.seekg(0, std::ios::beg);
    cipher_stream.seekg(0, std::ios::end);
    int data_len = cipher_stream.tellg();
    cipher_stream.seekg(0, std::ios::beg);
    size_t pos = constant::MAGIC_NUMBER_LEN + constant::VERSION_LEN + constant::TAG_LEN;

    size_t cipher_len = data_len - pos;
    std::unique_ptr<unsigned char[]> model_cipher(new unsigned char[cipher_len]);
    cipher_stream.seekg(pos); // skip header
    cipher_stream.read(reinterpret_cast<char *>(model_cipher.get()), cipher_len);

    size_t plain_len = data_len - AES_GCM_TAG_LENGTH - pos;
    std::unique_ptr<unsigned char[]> model_plain(new unsigned char[plain_len]);

    int ret_decrypt_file = util::crypto::AesGcm::decrypt_aes_gcm(
            model_cipher.get(),
            cipher_len,
            reinterpret_cast<const unsigned char*>(aes_key.c_str()),
            reinterpret_cast<const unsigned char*>(aes_iv.c_str()),
            model_plain.get(),
            reinterpret_cast<int&>(plain_len));

    if (ret_decrypt_file != CODE_OK) {
        LOGD("[M]decrypt file failed, decrypt ret = %d", ret_decrypt_file);
        return ret_decrypt_file;
    }

    plain_stream.write(reinterpret_cast<const char*>(model_plain.get()), plain_len);

    return CODE_OK;
}