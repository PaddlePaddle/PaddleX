//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <sstream>

#include "encryption/include/paddle_model_encrypt.h"
#include "encryption/include/paddle_stream_decrypt.h"

int main() {
    std::string key_data = paddle_generate_random_key();
    std::cout << "key is:" << key_data << std::endl;
    std::istringstream isst(std::string("hello world !"));
    std::ostringstream osst;
    int enc_ret =  encrypt_stream(key_data, isst, osst);
    if (enc_ret != 0) {
        std::cout << "ERROR paddle_encrypt_stream" << enc_ret <<std::endl;
        return 0;
    }
    std::istringstream isst_cipher(osst.str());
    std::ostringstream osst_plain;
    int dec_ret = decrypt_stream(isst_cipher, osst_plain, key_data);
    if (dec_ret != 0) {
        std::cout << "ERROR decrypt_stream " << dec_ret <<std::endl;
        return 0;
    }

    std::cout << "data is:" << osst_plain.str() << std::endl;
    return 0;
}
