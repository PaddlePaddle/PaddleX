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

#include <iostream>

#include "encryption/util/include/crypto/aes_gcm.h"

namespace util {
namespace crypto {

int AesGcm::aes_gcm_key(const unsigned char* key, const unsigned char* iv,
                        EVP_CIPHER_CTX* e_ctx, EVP_CIPHER_CTX* d_ctx) {
  int ret = 0;
  if (e_ctx != NULL) {
    ret = EVP_EncryptInit_ex(e_ctx, EVP_aes_256_gcm(), NULL, NULL, NULL);
    if (ret != 1) {
      return -1;
    }
    ret = EVP_CIPHER_CTX_ctrl(e_ctx, EVP_CTRL_GCM_SET_IVLEN, AES_GCM_IV_LENGTH,
                              NULL);
    if (ret != 1) {
      return -2;
    }
    ret = EVP_EncryptInit_ex(e_ctx, NULL, NULL, key, iv);
    if (ret != 1) {
      return -3;
    }
  }
  // initial decrypt ctx
  if (d_ctx != NULL) {
    ret = EVP_DecryptInit_ex(d_ctx, EVP_aes_256_gcm(), NULL, NULL, NULL);
    if (!ret) {
      return -1;
    }
    ret = EVP_CIPHER_CTX_ctrl(d_ctx, EVP_CTRL_GCM_SET_IVLEN, AES_GCM_IV_LENGTH,
                              NULL);
    if (!ret) {
      return -2;
    }
    ret = EVP_DecryptInit_ex(d_ctx, NULL, NULL, key, iv);
    if (!ret) {
      return -3;
    }
  }
  return 0;
}

int AesGcm::aes_gcm_key(const std::string& key_hex, const std::string& iv_hex,
                        EVP_CIPHER_CTX* e_ctx, EVP_CIPHER_CTX* d_ctx) {
  // check key_hex and iv_hex length
  if (key_hex.length() != AES_GCM_KEY_LENGTH * 2 ||
      iv_hex.length() != AES_GCM_IV_LENGTH * 2) {
    return -4;
  }

  unsigned char key[AES_GCM_KEY_LENGTH];
  unsigned char iv[AES_GCM_IV_LENGTH];

  int ret = Basic::hex_to_byte(key_hex, key);
  if (ret < 0) {
    return -5;
  }
  ret = Basic::hex_to_byte(iv_hex, iv);
  if (ret < 0) {
    return -5;
  }
  return aes_gcm_key(key, iv, e_ctx, d_ctx);
}

int AesGcm::encrypt_aes_gcm(const unsigned char* plaintext, const int& len,
                            const unsigned char* key, const unsigned char* iv,
                            unsigned char* ciphertext, int& out_len) {
  EVP_CIPHER_CTX* ctx = NULL;
  int ret = 0;
  int update_len = 0;
  int ciphertext_len = 0;
  unsigned char tag_char[AES_GCM_TAG_LENGTH];

  if (!(ctx = EVP_CIPHER_CTX_new())) {
    return -1;
  }
  // initial context
  ret = aes_gcm_key(key, iv, ctx, NULL);
  if (ret) {
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }
  // encryption
  ret = EVP_EncryptUpdate(ctx, ciphertext, &update_len, plaintext, len);
  if (ret != 1) {
    EVP_CIPHER_CTX_free(ctx);
    return -2;
  }
  ciphertext_len = update_len;

  ret = EVP_EncryptFinal_ex(ctx, ciphertext + ciphertext_len, &update_len);
  if (1 != ret) {
    EVP_CIPHER_CTX_free(ctx);
    return -3;
  }
  ciphertext_len += update_len;

  // Get the tags for authentication
  ret = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, AES_GCM_TAG_LENGTH,
                            tag_char);
  if (1 != ret) {
    EVP_CIPHER_CTX_free(ctx);
    return -4;
  }

  EVP_CIPHER_CTX_free(ctx);

  // append the tags to the end of encryption text
  for (int i = 0; i < AES_GCM_TAG_LENGTH; ++i) {
    ciphertext[ciphertext_len + i] = tag_char[i];
  }
  out_len = ciphertext_len + AES_GCM_TAG_LENGTH;

  return 0;
}

int AesGcm::decrypt_aes_gcm(const unsigned char* ciphertext, const int& len,
                            const unsigned char* key, const unsigned char* iv,
                            unsigned char* plaintext, int& out_len) {
  EVP_CIPHER_CTX* ctx = NULL;
  int ret = 0;
  int update_len = 0;
  int cipher_len = 0;
  int plaintext_len = 0;
  unsigned char tag_char[AES_GCM_TAG_LENGTH];

  // get the tag at the end of ciphertext
  for (int i = 0; i < AES_GCM_TAG_LENGTH; ++i) {
    tag_char[i] = ciphertext[len - AES_GCM_TAG_LENGTH + i];
  }
  cipher_len = len - AES_GCM_TAG_LENGTH;

  // initial aes context
  if (!(ctx = EVP_CIPHER_CTX_new())) {
    return -1;
  }

  ret = aes_gcm_key(key, iv, NULL, ctx);
  if (ret) {
    EVP_CIPHER_CTX_free(ctx);
    return -1;
  }

  // decryption
  ret = EVP_DecryptUpdate(ctx, plaintext, &update_len, ciphertext, cipher_len);
  if (ret != 1) {
    EVP_CIPHER_CTX_free(ctx);
    return -2;
  }
  plaintext_len = update_len;

  // check if the tag is equal to the decrption tag
  ret = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, AES_GCM_TAG_LENGTH,
                            tag_char);
  if (!ret) {
    EVP_CIPHER_CTX_free(ctx);
    // decrption failed
    return -3;
  }

  ret = EVP_DecryptFinal_ex(ctx, plaintext + update_len, &update_len);
  if (ret <= 0) {
    EVP_CIPHER_CTX_free(ctx);
    return -4;
  }

  plaintext_len += update_len;

  EVP_CIPHER_CTX_free(ctx);

  out_len = plaintext_len;
  return 0;
}

}  // namespace crypto
}  // namespace util
