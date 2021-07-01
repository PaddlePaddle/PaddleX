#ifndef PADDLE_MODEL_PROTECT_UTIL_BASIC_H
#define PADDLE_MODEL_PROTECT_UTIL_BASIC_H

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

namespace util {
namespace crypto {

class Basic {
public:
    /**
     * \brief        byte to hex
     *
     * \note         byte to hex.
     *               
     *
     * \param in_byte  byte array(in)
     * \param len      byte array length(in)
     * \param out_hex  the hex string(in)
     * 
     *
     * \return         return  0 if successful
     */
    static int byte_to_hex(
        const unsigned char* in_byte,
        int len,
        std::string& out_hex);

    /**
     * \brief        hex to byte
     *
     * \note         hex to byte.
     *               
     *
     * \param in_hex    the hex string(in)
     * \param out_byte  byte array(out)
     *
     * \return         return  0 if successful
     *                        -1 invalid in_hex
     */
    static int hex_to_byte(
        const std::string& in_hex,
        unsigned char* out_byte);

    /**
     * \brief        get random char for length
     *
     * \note         get random char for length
     *               
     *
     * \param array     to be random(out)
     * \param len       array length(in)
     *
     * \return         return  0 if successful
     *                        -1 invalid parameters
     */
    static int random(
        unsigned char* random,
        int len);
};

}
} // namespace common
#endif // PADDLE_MODEL_PROTECT_UTIL_BASIC_H
