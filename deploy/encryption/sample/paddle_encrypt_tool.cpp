#include <iostream>
#include <cstring>
#include "model_code.h"
#include "paddle_model_encrypt.h"
#include "paddle/paddle_inference_api.h"

#define RESET           "\033[0m"
#define BOLD            "\033[1m"
#define BOLDGREEN       "\033[1m\033[32m"

void help() {
    std::cout << BOLD << "*** paddle_encrypt_tool Usage ***" << RESET << std::endl;

    std::cout << "[1]Help:" << std::endl;
    std::cout << "\t-h" << std::endl;
    std::cout << "[2]Generate random key and encrypt dir files" << std::endl;
    std::cout << "\t-model_dir\tmodel_dir_ori\t-save_dir\tencrypted_models" << std::endl;
    std::cout << "[3]Generate random key for encrypt file" << std::endl;
    std::cout << "\t-g" << std::endl;
    std::cout << "[4]Encrypt file:" << std::endl;
    std::cout << "\t-e\t-key\tkeydata\t-infile\tinfile\t-outfile\toutfile" << std::endl;
}

int main(int argc, char** argv) {

    switch (argc) {
        case 2:
            if (strcmp(argv[1], "-g") == 0) {
                std::cout << BOLD << "Generate key success: \n\t" << RESET << BOLDGREEN << paddle_generate_random_key()
                          << RESET << std::endl;
            } else {
                help();
            }
            break;
        case 5:
            if (strcmp(argv[1], "-model_dir") == 0 && strcmp(argv[3], "-save_dir") == 0) {
                std::string key_random = paddle_generate_random_key();
                std::cout << BOLD << "Output: " << "Encryption key: \n\t" << RESET << BOLDGREEN
                          << key_random << RESET << std::endl;
                int ret = paddle_encrypt_dir(key_random.c_str(), argv[2], argv[4]);
                switch (ret) {
                    case CODE_OK:
                        std::cout << "Success, Encrypt __model__, __params__ to " << argv[4] << "(dir) success!"
                                  << std::endl;
                        break;
                    case CODE_MODEL_FILE_NOT_EXIST:
                        std::cout << "Failed, errorcode = " << ret << ", could't find __model__(file) in " << argv[2]
                                  << std::endl;
                        break;
                    case CODE_MODEL_YML_FILE_NOT_EXIST:
                        std::cout << "Failed, errorcode = " << ret << ", could't find model.yml(file) in " << argv[2]
                                  << std::endl;
                        break;
                    case CODE_PARAMS_FILE_NOT_EXIST:
                        std::cout << "Failed, errorcode = " << ret << ", could't find __params__(file) in " << argv[2]
                                  << std::endl;
                        break;
                    case CODE_NOT_EXIST_DIR:
                        std::cout << "Failed, errorcode = " << ret << ", " << argv[2] << "(dir) not exist" << std::endl;
                        break;
                    case CODE_FILES_EMPTY_WITH_DIR:
                        std::cout << "Failed, errorcode = " << ret << ", could't find any files in " << argv[2]
                                  << std::endl;
                        break;
                    default:std::cout << "Failed, errorcode = " << ret << ", others" << std::endl;
                        break;
                }
            } else {
                help();
            }
            break;
        case 8:
            if (strcmp(argv[1], "-e") == 0 && strcmp(argv[2], "-key") == 0 && strcmp(argv[4], "-infile") == 0
                && strcmp(argv[6], "-outfile") == 0) {
                int ret_encrypt = paddle_encrypt_model(argv[3], argv[5], argv[7]);
                if (ret_encrypt == 0) {
                    std::cout << "Encrypt " << argv[5] << "(file) to " << argv[7] << "(file) success" << std::endl;
                } else {
                    std::cout << "Encrypt " << argv[5] << " failed, ret = " << ret_encrypt << std::endl;
                }
            } else {
                help();
            }
            break;
        default:help();
    }

    return 0;
}