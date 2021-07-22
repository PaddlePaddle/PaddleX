// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <string>
#include <vector>

#include "model_deploy/common/include/paddle_deploy.h"

PaddleDeploy::Model* model;

extern "C" __declspec(dllexport) void InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file)
{
	bool use_gpu = false;
	int gpu_id = 0;

	// create model
	model = PaddleDeploy::CreateModel(model_type);  //FLAGS_model_type

	// model init
	model->Init(cfg_file);

	// inference engine init
	PaddleDeploy::PaddleEngineConfig engine_config;
	engine_config.model_filename = model_filename;
	engine_config.params_filename = params_filename;
	engine_config.use_gpu = use_gpu;
	engine_config.gpu_id = gpu_id;
	bool init = model->PaddleEngineInit(engine_config);
	if (init)
	{
		std::cout << "init model success" << std::endl;
	}	
}
/*
* img: input for predicting.
* 
* nWidth: width of img.
* 
* nHeight: height of img.
* 
* nChannel: channel of img.
* 
* output: result of pridict ,include category_id£¬score£¬coordinate¡£
* 
* nBoxesNum£º number of box
* 
* LabelList: label list of result
*/
extern "C" __declspec(dllexport) void ModelPredict(const unsigned char* img, int nWidth, int nHeight,int nChannel, float* output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel==1)
	{
		nType = CV_8UC1;
	}
	else if (nChannel == 2)
	{
		nType = CV_8UC2;
	}
	else if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else if (nChannel == 4)
	{
		nType = CV_8UC4;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	bool pre = model->Predict(imgs, &results, 1);
	if (pre)
	{
		std::cout << "model predict success" << std::endl;
	}
	nBoxesNum[0] = results.size();
	std::string label ="";
	for (int num = 0; num < results.size(); num++)
	{
		//std::cout << "res: " << results[num] << std::endl;
		for (int i = 0; i < results[num].det_result->boxes.size(); i++)
		{
			//std::cout << "category: " << results[num].det_result->boxes[i].category << std::endl;
			label = label + results[num].det_result->boxes[i].category+ " ";
			// labelindex
			output[num * 6 + 0] = results[num].det_result->boxes[i].category_id;
			// score
			output[num * 6 + 1] = results[num].det_result->boxes[i].score;
			//// box
			output[num * 6 + 2] = results[num].det_result->boxes[i].coordinate[0];
			output[num * 6 + 3] = results[num].det_result->boxes[i].coordinate[1];
			output[num * 6 + 4] = results[num].det_result->boxes[i].coordinate[2];
			output[num * 6 + 5] = results[num].det_result->boxes[i].coordinate[3];						
		}
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}

extern "C" __declspec(dllexport) void DestructModel()
{
	delete model;
	std::cout << "destruct model success" << std::endl;

}

//DEFINE_string(model_filename, "", "Path of det inference model");
//DEFINE_string(params_filename, "", "Path of det inference params");
//DEFINE_string(cfg_file, "", "Path of yaml file");
//DEFINE_string(model_type, "", "model type");
//DEFINE_string(image, "", "Path of test image file");
//DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
//DEFINE_int32(gpu_id, 0, "GPU card id");
//
//int main(int argc, char** argv) {
//  // Parsing command-line
//  google::ParseCommandLineFlags(&argc, &argv, true);
//
//  // create model
//  PaddleDeploy::Model* model = PaddleDeploy::CreateModel(FLAGS_model_type);
//
//  // model init
//  model->Init(FLAGS_cfg_file);
//
//  // inference engine init
//  PaddleDeploy::PaddleEngineConfig engine_config;
//  engine_config.model_filename = FLAGS_model_filename;
//  engine_config.params_filename = FLAGS_params_filename;
//  engine_config.use_gpu = FLAGS_use_gpu;
//  engine_config.gpu_id = FLAGS_gpu_id;
//  model->PaddleEngineInit(engine_config);
//
//  // prepare data
//  std::vector<cv::Mat> imgs;
//  imgs.push_back(std::move(cv::imread(FLAGS_image)));
//
//  // predict
//  std::vector<PaddleDeploy::Result> results;
//  model->Predict(imgs, &results, 1);
//
//  std::cout << results[0] << std::endl;
//  delete model;
//  return 0;
//}
