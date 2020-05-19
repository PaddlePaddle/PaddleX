//
// Created by zhongjiafeng on 5/12/20.
//

#include <vector>
#include <string>

#ifndef PADDLE_MODEL_PROTECT_UTIL_CRYPTO_SHA256_UTILS_H
#define PADDLE_MODEL_PROTECT_UTIL_CRYPTO_SHA256_UTILS_H

namespace util {
namespace crypto {

class SHA256Utils {
public:
    static void sha256(const void* data, size_t len, unsigned char* md);
    static std::vector<unsigned char> sha256(const void* data, size_t len);
    static std::vector<unsigned char> sha256(const std::vector<unsigned char>& data);
    static std::string sha256_string(const void* data, size_t len);
    static std::string sha256_string(const std::vector<unsigned char>& data);
    static std::string sha256_string(const std::string& string);
    static std::string sha256_file(const std::string& path);
};

}
}
#endif //PADDLE_MODEL_PROTECT_UTIL_CRYPTO_SHA256_UTILS_H
