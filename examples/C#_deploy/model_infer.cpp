#include <gflags/gflags.h>
#include <string>
#include <vector>

#include "model_deploy/common/include/paddle_deploy.h"

PaddleDeploy::Model* model;

// paddlex
extern "C" __declspec(dllexport) void InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, char* paddlex_model_type)
{
	// bool use_gpu = false;
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

	// det, seg, clas, paddlex
	if (strcmp(model_type, "paddlex") == 0) // 是paddlex模型，则返回具体支持的模型类型: det, seg, clas
	{
		// detector
		if (model->yaml_config_["model_type"].as<std::string>() == std::string("detector"))
		{
			strcpy(paddlex_model_type, "det");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("segmenter"))
		{
			strcpy(paddlex_model_type, "seg");
		}
		else if (model->yaml_config_["model_type"].as<std::string>() == std::string("classifier"))
		{
			strcpy(paddlex_model_type, "clas");
		}
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
extern "C" __declspec(dllexport) void Det_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 1)
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
		std::cout << "det model predict success" << std::endl;
	}
	// nBoxesNum[0] = results.size();  // results.size()得到的是batch_size
	nBoxesNum[0] = results[0].det_result->boxes.size();  // 得到单张图片预测的bounding box数
	std::string label = "";
	//std::cout << "res: " << results[num] << std::endl;
	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // 得到所有框的数据
	{
		//std::cout << "category: " << results[num].det_result->boxes[i].category << std::endl;
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // 类别的id
		// score
		output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // 得分
		//// box
		output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // 左上、右下的顶点
		output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


extern "C" __declspec(dllexport) void Seg_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 1)
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
		std::cout << "seg model predict success" << std::endl;
	}
	std::vector<uint8_t> result_map = results[0].seg_result->label_map.data; // vector<uint8_t> -- 结果map
	// 拷贝输出结果到输出上返回 -- 将vector<uint8_t>转成unsigned char *
	memcpy(output, &result_map[0], result_map.size() * sizeof(uchar));
}


extern "C" __declspec(dllexport) void Cls_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 1)
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
		std::cout << "cls model predict success" << std::endl;
	}
	*category_id = results[0].clas_result->category_id;
	// 拷贝输出类别结果到输出上返回 -- string --> char*
	memcpy(category, results[0].clas_result->category.c_str(), strlen(results[0].clas_result->category.c_str()));
	// 拷贝输出概率值返回
	*score = results[0].clas_result->score;
}



extern "C" __declspec(dllexport) void DestructModel()
{
	delete model;
	std::cout << "destruct model success" << std::endl;

}
