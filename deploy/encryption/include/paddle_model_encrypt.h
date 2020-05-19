#include <iostream>

#ifndef PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_ENCRYPT_H
#define PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_ENCRYPT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * generate random key
 * 产生随机的 key 信息，如果想要使用当前 SDK，
 * 对于传入的key信息有要求（需符合产生32字节随机值后做 BASE64 编码
 * @return
 */
std::string paddle_generate_random_key();

/**
 * encrypt __model__, __params__ files in src_dir to dst_dir
 * @param keydata
 * @param src_dir
 * @param dst_dir
 * @return
 */
int paddle_encrypt_dir(const char* keydata, const char* src_dir, const char* dst_dir);

/**
 * encrypt file
 * @param keydata   可使用由 paddle_generate_random_key 接口产生的key，也可以根据规则自己生成
 * @param infile
 * @param outfile
 * @return          error_code
 */
int paddle_encrypt_model(const char* keydata, const char* infile, const char* outfile);

#ifdef __cplusplus
}
#endif

#endif //PADDLE_MODEL_PROTECT_API_PADDLE_MODEL_ENCRYPT_H
