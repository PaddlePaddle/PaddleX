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
#include <stdio.h>
#include <gflags/gflags.h>
#include <iostream>
#include <string>

#include "encryption/include/model_code.h"
#include "encryption/include/paddle_model_encrypt.h"
#include "encryption/util/include/io_utils.h"

DEFINE_string(model_filename, "", "Path of model");
DEFINE_string(params_filename, "", "Path of params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(save_dir, "", "Path of save");
DEFINE_string(key, "", "encrypt key");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  if ("" == FLAGS_key) {
    FLAGS_key = paddle_generate_random_key();
  }

  if ("" == FLAGS_save_dir) {
      std::cerr << "Please input a save path" << std::endl;
      return -1;
  }
  int ret = ioutil::dir_exist_or_mkdir(FLAGS_save_dir.c_str());
  if (FLAGS_save_dir[FLAGS_save_dir.length() - 1] != '/') {
    FLAGS_save_dir.append("/");
  }

  std::string save_name[] = {"encrypted.yml",
                              "encrypted.pdmodel",
                              "encrypted.pdparams"};

  std::string input_files[] = {FLAGS_cfg_file,
                               FLAGS_model_filename,
                               FLAGS_params_filename};
  std::string outfile;
  for (auto i = 0; i < 3; ++i) {
    outfile = FLAGS_save_dir + save_name[i];
    ret = paddle_encrypt_model(FLAGS_key.c_str(),
                               input_files[i].c_str(),
                               outfile.c_str());
    if (ret != 0) {
      std::cerr << ret << ", Failed encrypt "
                << input_files[i] << std::endl;
      return -1;
    }
  }
  std::cout << "encrypt to " << FLAGS_save_dir << std::endl;
  return 0;
}
