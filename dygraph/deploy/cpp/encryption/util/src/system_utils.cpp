// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <sys/timeb.h>
#include <string.h>
#include <model_code.h>
#include <algorithm>
#include <iterator>

#include "encryption/util/include/system_utils.h"
#include "encryption/util/include/crypto/basic.h"
#include "encryption/util/include/crypto/sha256_utils.h"
#include "encryption/util/include/io_utils.h"
#include "encryption/util/include/log.h"
#include "encryption/util/include/constant/constant_model.h"

const char alphabet[] =
    "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(){}[]<>?~";

namespace util {

int SystemUtils::intN(int n) { return rand() % n; }

std::string SystemUtils::random_key_iv(int len) {
  unsigned char* tmp = (unsigned char*)malloc(sizeof(unsigned char) * len);
  int ret = util::crypto::Basic::random(tmp, len);
  std::string tmp_str(reinterpret_cast<const char*>(tmp), len);
  free(tmp);
  return tmp_str;
}

std::string SystemUtils::random_str(int len) {
  unsigned char* tmp = (unsigned char*)malloc(sizeof(unsigned char) * len);
  int ret = util::crypto::Basic::random(tmp, len);
  std::string tmp_str(reinterpret_cast<const char*>(tmp), len);
  free(tmp);
  return tmp_str;
}

int SystemUtils::check_key_match(const char* key, const char* filepath) {
  std::string aes_key_iv(key);
  std::string sha256_aes_key_iv =
      util::crypto::SHA256Utils::sha256_string(aes_key_iv);

  unsigned char* data_pos = (unsigned char*)malloc(sizeof(unsigned char) * 64);
  int ret = ioutil::read_with_pos_and_length(
      filepath, data_pos, constant::MAGIC_NUMBER_LEN + constant::VERSION_LEN,
      64);
  if (ret != CODE_OK) {
    LOGD("[M]read file failed when check key");
    return ret;
  }
  std::string check_str(reinterpret_cast<char*>(data_pos), 64);
  free(data_pos);

  if (strcmp(sha256_aes_key_iv.c_str(), check_str.c_str()) != 0) {
    return CODE_KEY_NOT_MATCH;
  }
  return CODE_OK;
}

int SystemUtils::check_key_match(const std::string& key,
                                 std::istream& cipher_stream) {
  cipher_stream.seekg(0, std::ios::beg);
  std::string sha256_aes_key_iv = util::crypto::SHA256Utils::sha256_string(key);
  int check_len = 64;

  std::string data_pos_str;
  cipher_stream.seekg(constant::MAGIC_NUMBER_LEN + constant::VERSION_LEN);
  std::copy_n(std::istreambuf_iterator<char>(cipher_stream), check_len,
              std::back_inserter(data_pos_str));
  if (data_pos_str.size() != check_len) {
    LOGD("[M]read file failed when check key");
    return CODE_OPEN_FAILED;
  }
  if (data_pos_str == sha256_aes_key_iv) {
    return CODE_OK;
  }

  return CODE_KEY_NOT_MATCH;
}

/**
 *
 * @param filepath
 * @return 0 - file encrypted    1 - file unencrypted
 */
int SystemUtils::check_file_encrypted(const char* filepath) {
  size_t read_len = constant::MAGIC_NUMBER_LEN + constant::VERSION_LEN;
  unsigned char* data_pos =
      (unsigned char*)malloc(sizeof(unsigned char) * read_len);
  if (ioutil::read_with_pos_and_length(filepath, data_pos, 0, read_len) !=
      CODE_OK) {
    LOGD("check file failed when read %s(file)", filepath);
    return CODE_OPEN_FAILED;
  }

  std::string tag(constant::MAGIC_NUMBER);
  tag.append(constant::VERSION);
  int ret_cmp = strcmp(tag.c_str(), (const char*)data_pos) == 0 ? 0 : 1;
  free(data_pos);
  return ret_cmp;
}

int SystemUtils::check_file_encrypted(std::istream& cipher_stream) {
  cipher_stream.seekg(0, std::ios::beg);
  size_t read_len = constant::MAGIC_NUMBER_LEN + constant::VERSION_LEN;
  std::string data_pos_str;
  std::copy_n(std::istreambuf_iterator<char>(cipher_stream), read_len,
              std::back_inserter(data_pos_str));
  if (data_pos_str.size() != read_len) {
    LOGD("check file failed when read cipher stream");
    return CODE_OPEN_FAILED;
  }

  std::string tag(constant::MAGIC_NUMBER);
  tag.append(constant::VERSION);
  if (data_pos_str == tag) {
    return 0;
  }

  return 1;
}

int SystemUtils::check_pattern_exist(const std::vector<std::string>& vecs,
                                     const std::string& pattern) {
  if (std::find(vecs.begin(), vecs.end(), pattern) == vecs.end()) {
    return -1;  // not exist
  } else {
    return 0;  // exist
  }
}

}  // namespace util
