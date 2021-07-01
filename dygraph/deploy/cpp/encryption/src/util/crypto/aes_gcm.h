#ifndef PADDLE_MODEL_PROTECT_UTIL_CRYPTO_AES_GCM_H
#define PADDLE_MODEL_PROTECT_UTIL_CRYPTO_AES_GCM_H

#include <iostream>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <string>

#include "basic.h"

namespace util {
namespace crypto {
// aes key 32 byte for 256 bit
#define AES_GCM_KEY_LENGTH 32

// aes tag 16 byte for 128 bit
#define AES_GCM_TAG_LENGTH 16

// aes iv 12 byte for 96 bit
#define AES_GCM_IV_LENGTH 16

class AesGcm {

public:
    /**
     * \brief        initial aes-gcm-256 context use key & iv
     *
     * \note         initial aes-gcm-256 context use key & iv. gcm mode
     *               will generate a tag(16 byte), so the ciphertext's length
     *               should be longer 16 byte than plaintext.
     *               
     *
     * \param plaintext     plain text to be encrypted(in)
     * \param len      plain text's length(in)
     * \param key     aes key (in)
     * \param iv      aes iv (in)
     * \param ciphertext   encrypted text(out)
     * \param out_len   encrypted length(out)
     *
     * \return         return  0 if successful
     *                        -1 EVP_CIPHER_CTX_new or aes_gcm_key error
     *                        -2 EVP_EncryptUpdate error
     *                        -3 EVP_EncryptFinal_ex error
     *                        -4 EVP_CIPHER_CTX_ctrl error
     */
    static int encrypt_aes_gcm(
        const unsigned char* plaintext,
        const int& len,
        const unsigned char* key,
        const unsigned char* iv,
        unsigned char* ciphertext,
        int& out_len);
    /**
     * \brief        encrypt using aes-gcm-256
     *
     * \note         encrypt using aes-gcm-256
     *
     * \param ciphertext     cipher text to be decrypted(in)
     * \param len      plain text's length(in)
     * \param key     aes key (in)
     * \param iv      aes iv (in)
     * \param plaintext   decrypted text(out)
     * \param out_len   decrypted length(out)
     *
     * \return         return  0 if successful
     *                        -1 EVP_CIPHER_CTX_new or aes_gcm_key error
     *                        -2 EVP_DecryptUpdate error
     *                        -3 EVP_CIPHER_CTX_ctrl error
     *                        -4 EVP_DecryptFinal_ex error
     */
    static int decrypt_aes_gcm(
        const unsigned char* ciphertext,
        const int& len,
        const unsigned char* key,
        const unsigned char* iv,
        unsigned char* plaintext,
        int& out_len);

private:
    /**
     * \brief        initial aes-gcm-256 context use key & iv
     *
     * \note         initial aes-gcm-256 context use key & iv
     *
     * \param key     aes key (in)
     * \param iv      aes iv (in)
     * \param e_ctx   encryption context(out)
     * \param d_ctx   decryption context(out)
     *
     * \return         return  0 if successful
     *                        -1 EVP_xxcryptInit_ex error
     *                        -2 EVP_CIPHER_CTX_ctrl error
     *                        -3 EVP_xxcryptInit_ex error
     */
    static int aes_gcm_key(
        const unsigned char* key,
        const unsigned char* iv,
        EVP_CIPHER_CTX* e_ctx,
        EVP_CIPHER_CTX* d_ctx);

    /**
     * \brief        initial aes-gcm-256 context use key & iv
     *
     * \note         initial aes-gcm-256 context use key & iv
     *
     * \param key     aes key (in)
     * \param iv      aes iv (in)
     * \param e_ctx   encryption context(out)
     * \param d_ctx   decryption context(out)
     *
     * \return         return  0 if successful
     *                        -1 EVP_xxcryptInit_ex error
     *                        -2 EVP_CIPHER_CTX_ctrl error
     *                        -3 EVP_xxcryptInit_ex error
     *                        -4 invalid key length or iv length
     *                        -5 hex_to_byte error
     */
    static int aes_gcm_key(
        const std::string& key_hex,
        const std::string& iv_hex,
        EVP_CIPHER_CTX* e_ctx,
        EVP_CIPHER_CTX* d_ctx);
};

} // namespace crypt
} // namespace common

#endif // PADDLE_MODEL_PROTECT_UTIL_CRYPTO_AES_GCM_H