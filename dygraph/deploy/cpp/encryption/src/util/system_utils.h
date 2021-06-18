#include <string>
#include <vector>

#ifndef PADDLE_MODEL_PROTECT_SYSTEM_UTIL_H
#define PADDLE_MODEL_PROTECT_SYSTEM_UTIL_H

namespace util {

class SystemUtils {
public:
    static std::string random_key_iv(int len);
    static std::string random_str(int len);
    static int check_key_match(const char* key, const char* filepath);
    static int check_key_match(const std::string &key, std::istream &cipher_stream);
    static int check_file_encrypted(const char* filepath);
    static int check_file_encrypted(std::istream &cipher_stream);
    static int check_pattern_exist(const std::vector<std::string>& vecs, const std::string& pattern);

private:
    inline static int intN(int n);

};

}

#endif //PADDLE_MODEL_PROTECT_SYSTEM_UTIL_H
