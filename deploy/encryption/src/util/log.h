#ifndef PADDLE_MODEL_PROTECT_UTIL_LOG_H
#define PADDLE_MODEL_PROTECT_UTIL_LOG_H

#include <stdio.h>
#include <unistd.h>

#define LOGD(fmt,...)\
    printf("{%s:%u}:" fmt "\n", __FUNCTION__, __LINE__,  ##__VA_ARGS__)

#endif //PADDLE_MODEL_PROTECT_UTIL_LOG_H
