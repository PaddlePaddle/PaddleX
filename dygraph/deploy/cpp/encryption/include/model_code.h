#ifndef PADDLE_MODEL_PROTECT_MODEL_CODE_H
#define PADDLE_MODEL_PROTECT_MODEL_CODE_H

#ifdef __cplusplus
extern  "C" {
#endif

    enum  {
        CODE_OK                         = 0,
        CODE_OPEN_FAILED                = 100,
        CODE_READ_FILE_PTR_IS_NULL      = 101,
        CODE_AES_GCM_ENCRYPT_FIALED     = 102,
        CODE_AES_GCM_DECRYPT_FIALED     = 103,
        CODE_KEY_NOT_MATCH              = 104,
        CODE_KEY_LENGTH_ABNORMAL        = 105,
        CODE_NOT_EXIST_DIR              = 106,
        CODE_FILES_EMPTY_WITH_DIR       = 107,
        CODE_MODEL_FILE_NOT_EXIST       = 108,
        CODE_PARAMS_FILE_NOT_EXIST      = 109,
        CODE_MODEL_YML_FILE_NOT_EXIST   = 110,
        CODE_MKDIR_FAILED               = 111
    };


#ifdef __cplusplus
}
#endif

#endif //PADDLE_MODEL_PROTECT_MODEL_CODE_H
