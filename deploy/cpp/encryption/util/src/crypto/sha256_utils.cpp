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
#include <stdio.h>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>

#include "encryption/util/include/crypto/sha256_utils.h"

namespace util {
namespace crypto {

void SHA256Utils::sha256(const void* data, size_t len, unsigned char* md) {
  SHA256_CTX sha_ctx = {};
  SHA256_Init(&sha_ctx);
  SHA256_Update(&sha_ctx, data, len);
  SHA256_Final(md, &sha_ctx);
}
std::vector<unsigned char> SHA256Utils::sha256(const void* data, size_t len) {
  std::vector<unsigned char> md(32);
  sha256(data, len, &md[0]);
  return md;
}
std::vector<unsigned char> SHA256Utils::sha256(
    const std::vector<unsigned char>& data) {
  return sha256(&data[0], data.size());
}
std::string SHA256Utils::sha256_string(const void* data, size_t len) {
  std::vector<unsigned char> md = sha256(data, len);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (unsigned char c : md) {
    oss << std::setw(2) << int(c);
  }
  return oss.str();
}
std::string SHA256Utils::sha256_string(const std::vector<unsigned char>& data) {
  return sha256_string(&data[0], data.size());
}
std::string SHA256Utils::sha256_string(const std::string& string) {
  return sha256_string(string.c_str(), string.size());
}
std::string SHA256Utils::sha256_file(const std::string& path) {
  FILE* file = fopen(path.c_str(), "rb");
  if (!file) {
    return "";
  }
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_CTX sha_ctx = {};
  SHA256_Init(&sha_ctx);
  const int size = 32768;
  void* buffer = malloc(size);
  if (!buffer) {
    fclose(file);
    return "";
  }
  int read = 0;
  while ((read = fread(buffer, 1, size, file))) {
    SHA256_Update(&sha_ctx, buffer, read);
  }
  SHA256_Final(hash, &sha_ctx);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (unsigned char c : hash) {
    oss << std::setw(2) << int(c);
  }
  fclose(file);
  free(buffer);
  return oss.str();
}

}  // namespace crypto
}  // namespace util
