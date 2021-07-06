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

#include "encryption/util/include/crypto/base64.h"

using std::string;

namespace baidu {
namespace base {
namespace base64 {

namespace {
const string base64_chars =  // NOLINT
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

inline bool is_base64(unsigned char c) {
  return isalnum(c) || (c == '+') || (c == '/');
}

inline size_t encode_len(size_t input_len) { return (input_len + 2) / 3 * 4; }

void encode_char_array(unsigned char *encode_block,
                       const unsigned char *decode_block) {
  encode_block[0] = (decode_block[0] & 0xfc) >> 2;
  encode_block[1] =
      ((decode_block[0] & 0x03) << 4) + ((decode_block[1] & 0xf0) >> 4);
  encode_block[2] =
      ((decode_block[1] & 0x0f) << 2) + ((decode_block[2] & 0xc0) >> 6);
  encode_block[3] = decode_block[2] & 0x3f;
}

void decode_char_array(unsigned char *encode_block,
                       unsigned char *decode_block) {
  for (int i = 0; i < 4; ++i) {
    encode_block[i] = base64_chars.find(encode_block[i]);
  }
  decode_block[0] = (encode_block[0] << 2) + ((encode_block[1] & 0x30) >> 4);
  decode_block[1] =
      ((encode_block[1] & 0xf) << 4) + ((encode_block[2] & 0x3c) >> 2);
  decode_block[2] = ((encode_block[2] & 0x3) << 6) + encode_block[3];
}
}  // namespace

string base64_encode(const string &input) {
  string output;
  size_t i = 0;
  unsigned char decode_block[3];
  unsigned char encode_block[4];

  for (string::size_type len = 0; len != input.size(); ++len) {
    decode_block[i++] = input[len];
    if (i == 3) {
      encode_char_array(encode_block, decode_block);
      for (i = 0; i < 4; ++i) {
        output += base64_chars[encode_block[i]];
      }
      i = 0;
    }
  }

  if (i > 0) {
    for (size_t j = i; j < 3; ++j) {
      decode_block[j] = '\0';
    }

    encode_char_array(encode_block, decode_block);

    for (size_t j = 0; j < i + 1; ++j) {
      output += base64_chars[encode_block[j]];
    }

    while (i++ < 3) {
      output += '=';
    }
  }

  return output;
}

string base64_decode(const string &encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int len = 0;
  unsigned char encode_block[4];
  unsigned char decode_block[3];
  string output;

  while (in_len-- && (encoded_string[len] != '=') &&
         is_base64(encoded_string[len])) {
    encode_block[i++] = encoded_string[len];
    len++;
    if (i == 4) {
      decode_char_array(encode_block, decode_block);

      for (int j = 0; j < 3; ++j) {
        output += decode_block[j];
      }
      i = 0;
    }
  }

  if (i > 0) {
    for (int j = i; j < 4; ++j) {
      encode_block[j] = 0;
    }

    decode_char_array(encode_block, decode_block);

    for (int j = 0; j < i - 1; ++j) {
      output += decode_block[j];
    }
  }

  return output;
}

}  // namespace base64
}  // namespace base
}  // namespace baidu
// #include <string>
// #include <cassert>
// #include <limits>
// #include <stdexcept>
// #include <ctype.h>
// #include "base64_utils.h"

// namespace util {
// namespace crypto {

// static const char b64_table[65] =
// "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// static const char reverse_table[128] = {
//     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
//     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
//     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
//     52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
//     64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
//     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
//     64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
//     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64
// };

// std::string Base64Utils::encode(const ::std::string& data) {
//     try {
//         if (data.size() > (std::numeric_limits<std::string::size_type>::max()
//         / 4u) * 3u) {
//             throw ::std::length_error("Converting too large a string to
//             base64.");
//         }
//     } catch (std::length_error& e) {
//         printf("%s\n", e.what());
//         return "";
//     }

//     const std::size_t binlen = data.size();
//     std::string retval((((binlen + 2) / 3) * 4), '=');
//     std::size_t outpos = 0;
//     int bits_collected = 0;
//     unsigned int accumulator = 0;
//     const std::string::const_iterator binend = data.end();

//     for (std::string::const_iterator i = data.begin(); i != binend; ++i) {
//         accumulator = (accumulator << 8) | (*i & 0xffu);
//         bits_collected += 8;
//         while (bits_collected >= 6) {
//             bits_collected -= 6;
//             retval[outpos++] = b64_table[(accumulator >> bits_collected) &
//             0x3fu];
//         }
//     }
//     if (bits_collected > 0) { // Any trailing bits that are missing.
//         assert(bits_collected < 6);
//         accumulator <<= 6 - bits_collected;
//         retval[outpos++] = b64_table[accumulator & 0x3fu];
//     }
//     assert(outpos >= (retval.size() - 2));
//     assert(outpos <= retval.size());
//     return retval;
// }

// std::string Base64Utils::decode(const std::string& data) {
//     std::string retval;
//     const std::string::const_iterator last = data.end();
//     int bits_collected = 0;
//     unsigned int accumulator = 0;

//     try {

//         for (std::string::const_iterator i = data.begin(); i != last; ++i) {
//             const int c = *i;
//             if (isspace(c) || c == '=') {
//                 continue;
//             }
//             if ((c > 127) || (c < 0) || (reverse_table[c] > 63)) {
//                 throw ::std::invalid_argument("This contains characters not
//                 legal in a base64 encoded string.");
//             }
//             accumulator = (accumulator << 6) | reverse_table[c];
//             bits_collected += 6;
//             if (bits_collected >= 8) {
//                 bits_collected -= 8;
//                 retval += static_cast<char>((accumulator >> bits_collected) &
//                 0xffu);
//             }
//         }
//     } catch (std::invalid_argument& e) {
//         printf("%s\n", e.what());
//         return "";
//     }

//     return retval;
// }

// }
// }
