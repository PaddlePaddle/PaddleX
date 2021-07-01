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
#include "encryption/util/include/crypto/basic.h"

namespace util {
namespace crypto {

int Basic::byte_to_hex(const unsigned char* in_byte, int len,
                       std::string& out_hex) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < len; ++i) {
    oss << std::setw(2) << int(in_byte[i]);
  }
  out_hex = oss.str();
  return 0;
}

int Basic::hex_to_byte(const std::string& in_hex, unsigned char* out_byte) {
  int i = 0;
  int j = 0;
  int len = in_hex.length() / 2;
  const unsigned char* hex;
  if (in_hex.length() % 2 != 0 || out_byte == NULL) {
    return -1;
  }
  hex = (unsigned char*)in_hex.c_str();

  for (; j < len; i += 2, ++j) {
    unsigned char high = hex[i];
    unsigned char low = hex[i + 1];
    if (high >= '0' && high <= '9') {
      high = high - '0';
    } else if (high >= 'A' && high <= 'F') {
      high = high - 'A' + 10;
    } else if (high >= 'a' && high <= 'f') {
      high = high - 'a' + 10;
    } else {
      return -2;
    }

    if (low >= '0' && low <= '9') {
      low = low - '0';
    } else if (low >= 'A' && low <= 'F') {
      low = low - 'A' + 10;
    } else if (low >= 'a' && low <= 'f') {
      low = low - 'a' + 10;
    } else {
      return -2;
    }
    out_byte[j] = high << 4 | low;
  }
  return 0;
}

int Basic::random(unsigned char* random, int len) {
  std::random_device rd;
  int i = 0;
  if (len <= 0 || random == NULL) {
    return -1;
  }
  for (; i < len; ++i) {
    random[i] = rd() % 256;
  }
  return 0;
}

}  // namespace crypto
}  // namespace util
