#ifndef PADDLE_MODEL_PROTECT_CONSTANT_CONSTANT_MODEL_H
#define PADDLE_MODEL_PROTECT_CONSTANT_CONSTANT_MODEL_H

namespace constant {

const static std::string MAGIC_NUMBER = "PADDLE";
const static std::string VERSION = "1";

const static int MAGIC_NUMBER_LEN = 6;
const static int VERSION_LEN = 1;
const static int TAG_LEN = 128;

const static int CONSTANT_AES_GCM_KEY_LEN = 32;

}



#endif //PADDLE_MODEL_PROTECT_CONSTANT_CONSTANT_MODEL_H
