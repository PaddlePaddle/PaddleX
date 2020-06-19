#include "base64.h"

using std::string;

namespace baidu {
namespace base {
namespace base64 {

namespace {
const string base64_chars = 
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "abcdefghijklmnopqrstuvwxyz"
         "0123456789+/";

inline bool is_base64(unsigned char c) {
    return isalnum(c) || (c == '+') || (c == '/');
}

inline size_t encode_len(size_t input_len) {
    return (input_len + 2) / 3 * 4;
}

void encode_char_array(unsigned char *encode_block, const unsigned char *decode_block) {
    encode_block[0] = (decode_block[0] & 0xfc) >> 2;
    encode_block[1] = ((decode_block[0] & 0x03) << 4) + ((decode_block[1] & 0xf0) >> 4);
    encode_block[2] = ((decode_block[1] & 0x0f) << 2) + ((decode_block[2] & 0xc0) >> 6);
    encode_block[3] = decode_block[2] & 0x3f;
}

void decode_char_array(unsigned char *encode_block, unsigned char *decode_block) {
    for (int i = 0; i < 4; ++i) {
        encode_block[i] = base64_chars.find(encode_block[i]);
    }
    decode_block[0] = (encode_block[0] << 2) + ((encode_block[1] & 0x30) >> 4);
    decode_block[1] = ((encode_block[1] & 0xf) << 4) + ((encode_block[2] & 0x3c) >> 2);
    decode_block[2] = ((encode_block[2] & 0x3) << 6) + encode_block[3];
}
}

string base64_encode(const string& input) {
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

string base64_decode(const string& encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int len = 0;
    unsigned char encode_block[4];
    unsigned char decode_block[3];
    string output;

    while (in_len-- && (encoded_string[len] != '=') && is_base64(encoded_string[len])) {
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

}
}
}