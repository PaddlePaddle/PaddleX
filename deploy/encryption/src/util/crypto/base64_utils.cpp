#include <string>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <ctype.h>
#include "base64_utils.h"

namespace util {
namespace crypto {

static const char b64_table[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static const char reverse_table[128] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
    64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64
};

std::string Base64Utils::encode(const ::std::string& data) {
    try {
        if (data.size() > (std::numeric_limits<std::string::size_type>::max() / 4u) * 3u) {
            throw ::std::length_error("Converting too large a string to base64.");
        }
    } catch (std::length_error& e) {
        printf("%s\n", e.what());
        return "";
    }

    const std::size_t binlen = data.size();
    std::string retval((((binlen + 2) / 3) * 4), '=');
    std::size_t outpos = 0;
    int bits_collected = 0;
    unsigned int accumulator = 0;
    const std::string::const_iterator binend = data.end();

    for (std::string::const_iterator i = data.begin(); i != binend; ++i) {
        accumulator = (accumulator << 8) | (*i & 0xffu);
        bits_collected += 8;
        while (bits_collected >= 6) {
            bits_collected -= 6;
            retval[outpos++] = b64_table[(accumulator >> bits_collected) & 0x3fu];
        }
    }
    if (bits_collected > 0) { // Any trailing bits that are missing.
        assert(bits_collected < 6);
        accumulator <<= 6 - bits_collected;
        retval[outpos++] = b64_table[accumulator & 0x3fu];
    }
    assert(outpos >= (retval.size() - 2));
    assert(outpos <= retval.size());
    return retval;
}

std::string Base64Utils::decode(const std::string& data) {
    std::string retval;
    const std::string::const_iterator last = data.end();
    int bits_collected = 0;
    unsigned int accumulator = 0;

    try {

        for (std::string::const_iterator i = data.begin(); i != last; ++i) {
            const int c = *i;
            if (isspace(c) || c == '=') {
                continue;
            }
            if ((c > 127) || (c < 0) || (reverse_table[c] > 63)) {
                throw ::std::invalid_argument("This contains characters not legal in a base64 encoded string.");
            }
            accumulator = (accumulator << 6) | reverse_table[c];
            bits_collected += 6;
            if (bits_collected >= 8) {
                bits_collected -= 8;
                retval += static_cast<char>((accumulator >> bits_collected) & 0xffu);
            }
        }
    } catch (std::invalid_argument& e) {
        printf("%s\n", e.what());
        return "";
    }

    return retval;
}

}
}


