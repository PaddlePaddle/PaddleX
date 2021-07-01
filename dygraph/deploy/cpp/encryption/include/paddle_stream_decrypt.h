#pragma once

#include <stdio.h>

#ifndef PADDLE_MODEL_PROTECT_API_PADDLE_STREAM_DECRYPT_H
#define PADDLE_MODEL_PROTECT_API_PADDLE_STREAM_DECRYPT_H

#ifdef WIN32
#ifdef PM_EXPORTS
#define PDD_MODEL_API __declspec(dllexport)
#else
#define PDD_MODEL_API __declspec(dllimport)
#endif
#endif
#ifdef linux
#define PDD_MODEL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * check file (un)encrypted?
 * @param file_path
 * @return
 */

PDD_MODEL_API int paddle_check_stream_encrypted(std::istream &cipher_stream);

PDD_MODEL_API int decrypt_stream(std::istream &cipher_stream, std::ostream &plain_stream, const std::string &key_base64);

#ifdef __cplusplus
}
#endif

#endif //PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_DECRYPT_H
