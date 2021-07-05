#include "../include/paddle_model_encrypt.h"
#include "../include/paddle_stream_decrypt.h"

#include <iostream>
#include <string>
#include <sstream>

int main(){

    std::string key_data = paddle_generate_random_key();
    std::cout << "key is:" << key_data << std::endl;
    std::istringstream isst(std::string("hello world !"));
    std::ostringstream osst;
    int enc_ret =  encrypt_stream(key_data, isst, osst);
    if (enc_ret != 0){
        std::cout << "ERROR paddle_encrypt_stream" << enc_ret <<std::endl;
        return 0;
    }
    std::istringstream isst_cipher(osst.str());
    std::ostringstream osst_plain;
    int dec_ret = decrypt_stream(isst_cipher, osst_plain, key_data);
    if (dec_ret != 0){
        std::cout << "ERROR decrypt_stream " << dec_ret <<std::endl;
        return 0;
    }

    std::cout << "data is:" << osst_plain.str() << std::endl;

    return 0;
}


    
