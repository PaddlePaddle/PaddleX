#include <vector>
#include <string>

#ifndef PADDLE_MODEL_PROTECT_UTIL_CRYPTO_BASE64_UTILS_H
#define PADDLE_MODEL_PROTECT_UTIL_CRYPTO_BASE64_UTILS_H

namespace baidu {
namespace base {
namespace base64 {

std::string base64_encode(const std::string& input);
std::string base64_decode(const std::string& input);

}
}
}

// namespace util {
// namespace crypto {

// class Base64Utils {
// public:
//     static std::string encode(const ::std::string& data);

//     static std::string decode(const ::std::string& data);
// };

// }
// }
#endif //PADDLE_MODEL_PROTECT_BASE64_UTILS_H
