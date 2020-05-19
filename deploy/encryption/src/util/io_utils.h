//
// Created by zhongjiafeng on 5/11/20.
//

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#ifndef PADDLE_MODEL_PROTECT_IO_UTILS_H
#define PADDLE_MODEL_PROTECT_IO_UTILS_H

namespace ioutil {

int read_file(const char* file_path, unsigned char** dataptr, size_t* sizeptr);

int read_with_pos_and_length(const char* file_path, unsigned char* dataptr, size_t pos, size_t length);

int read_with_pos(const char* file_path, size_t pos, unsigned char** dataptr, size_t* sizeptr);

int write_file(const char* file_path, const unsigned char* dataptr, size_t sizeptr);

int append_file(const char* file_path, const unsigned char* data, size_t len);

size_t read_file_size(const char* file_path);

int read_file_to_file(const char* src_path, const char* dst_path);

int dir_exist_or_mkdir(const char* dir);

/**
 * @return files.size()
 */
int read_dir_files(const char* dir_path, std::vector<std::string>& files);

}
#endif //PADDLE_MODEL_PROTECT_IO_UTILS_H
