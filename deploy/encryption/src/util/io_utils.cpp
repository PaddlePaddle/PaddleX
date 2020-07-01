#ifdef __linux__
#include <unistd.h>
#include <dirent.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#endif

#include <iostream>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "io_utils.h"
#include "model_code.h"
#include "log.h"

namespace ioutil {

int read_file(const char* file_path, unsigned char** dataptr, size_t* sizeptr) {
    FILE* fp = NULL;
    fp = fopen(file_path, "rb");
    if (fp == NULL) {
        LOGD("[M]open file(%s) failed", file_path);
        return CODE_OPEN_FAILED;
    }

    fseek(fp, 0, SEEK_END);
    *sizeptr = ftell(fp);
    *dataptr = (unsigned char*) malloc(sizeof(unsigned char) * (*sizeptr));

    fseek(fp, 0, SEEK_SET);
    fread(*dataptr, 1, *sizeptr, fp);
    fclose(fp);

    return CODE_OK;
}

int read_with_pos_and_length(const char* file_path, unsigned char* dataptr, size_t pos, size_t length) {
    if (dataptr == NULL) {
        LOGD("Read file pos dataptr = NULL");
        return CODE_READ_FILE_PTR_IS_NULL;
    }

    FILE* fp = NULL;
    if ((fp = fopen(file_path, "rb")) == NULL) {
        LOGD("[M]open file(%s) failed", file_path);
        return CODE_OPEN_FAILED;
    }

    fseek(fp, pos, SEEK_SET);
    fread(dataptr, 1, length, fp);
    fclose(fp);

    return CODE_OK;
}

int read_with_pos(const char* file_path, size_t pos, unsigned char** dataptr, size_t* sizeptr) {

    FILE* fp = NULL;
    if ((fp = fopen(file_path, "rb")) == NULL) {
        LOGD("[M]open file(%s) failed when read_with_pos", file_path);
        return CODE_OPEN_FAILED;
    }

    fseek(fp, 0, SEEK_END);
    size_t filesize = ftell(fp);

    *sizeptr = filesize - pos;
    *dataptr = (unsigned char*) malloc(sizeof(unsigned char) * (filesize - pos));
    fseek(fp, pos, SEEK_SET);
    fread(*dataptr, 1, filesize - pos, fp);
    fclose(fp);

    return CODE_OK;
}

int write_file(const char* file_path, const unsigned char* dataptr, size_t sizeptr) {
    FILE* fp = NULL;
    if ((fp = fopen(file_path, "wb")) == NULL) {
        LOGD("[M]open file(%s) failed", file_path);
        return CODE_OPEN_FAILED;
    }

    fwrite(dataptr, 1, sizeptr, fp);

    fclose(fp);
    return CODE_OK;
}

int append_file(const char* file_path, const unsigned char* data, size_t len) {
    FILE* fp = fopen(file_path, "ab+");
    if (fp == NULL) {
        LOGD("[M]open file(%s) failed when append_file", file_path);
        return CODE_OPEN_FAILED;
    }
    fwrite(data, sizeof(char), len, fp);
    fclose(fp);
    return CODE_OK;
}

size_t read_file_size(const char* file_path) {
    FILE* fp = NULL;
    fp = fopen(file_path, "rb");
    if (fp == NULL) {
        LOGD("[M]open file(%s) failed when read_file_size", file_path);
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    size_t filesize = ftell(fp);
    fclose(fp);

    return filesize;
}

int read_file_to_file(const char* src_path, const char* dst_path) {
    FILE* infp = NULL;
    if ((infp = fopen(src_path, "rb")) == NULL) {
        LOGD("[M]read src file failed when read_file_to_file");
        return CODE_OPEN_FAILED;
    }

    fseek(infp, 0, SEEK_END);
    size_t insize = ftell(infp);
    char* content = (char*) malloc(sizeof(char) * insize);

    fseek(infp, 0, SEEK_SET);
    fread(content, 1, insize, infp);
    fclose(infp);

    FILE* outfp = NULL;
    if ((outfp = fopen(dst_path, "wb")) == NULL) {
        LOGD("[M]open dst file failed when read_file_to_file");
        return CODE_OPEN_FAILED;
    }
    fwrite(content, 1, insize, outfp);
    fclose(outfp);
    free(content);
    return CODE_OK;
}

int read_dir_files(const char* dir_path, std::vector<std::string>& files) {
#ifdef __linux__
    struct dirent* ptr;
    DIR* dir = NULL;
    dir = opendir(dir_path);
    if (dir == NULL) {
        return -1; // CODE_NOT_EXIST_DIR
    }
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);
#endif
#ifdef _WIN32
    intptr_t handle;
	struct _finddata_t fileinfo;
	
	std::string tmp_dir(dir_path);
	std::string::size_type idx = tmp_dir.rfind("\\*");
	if (idx == std::string::npos || idx != tmp_dir.length() - 1)
	{
		tmp_dir.append("\\*");
	}

	handle = _findfirst(tmp_dir.c_str(), &fileinfo);
	if (handle == -1) {
		return -1;
	}

	do {
		std::cout << "File name = " << fileinfo.name << std::endl;
		if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
			files.push_back(fileinfo.name);
		}
	} while (!_findnext(handle, &fileinfo));

std::cout << files.size() << std::endl;
    for (size_t i = 0; i < files.size(); i++)
    {
        std::cout << files[i] << std::endl;
    }

	_findclose(handle);
#endif
    return files.size();
}

int dir_exist_or_mkdir(const char* dir) {
#ifdef _WIN32
    if (CreateDirectory(dir, NULL)) {
        // return CODE_OK;
    } else {
        return CODE_MKDIR_FAILED;
    }
    
#endif
#ifdef __linux__
    if (access(dir, 0) != 0) {
        mkdir(dir, S_IRWXU | S_IRWXG | S_IRWXO);
    }
#endif
    return CODE_OK;
}

}
