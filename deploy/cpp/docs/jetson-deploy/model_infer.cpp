#include <string>
#include <vector>

#include "model_deploy/common/include/paddle_deploy.h"


PaddleDeploy::Model* model;

/*
* 模型初始化/注册接口
*
* model_type: 初始化模型类型: det,seg,clas,paddlex
*
* model_filename: 模型文件路径
*
* params_filename: 参数文件路径
*
* cfg_file: 配置文件路径
*
* use_gpu: 是否使用GPU
*
* gpu_id: 指定第x号GPU
*
* paddlex_model_type: model_type为paddlx时，返回的实际paddlex模型的类型: det, seg, clas
*
*/
extern "C" void InitModel(const char* model_type, const char* model_filename, const char* params_filename, const char* cfg_file, bool use_gpu, int gpu_id, char* paddlex_model_type)
{
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
* 检测推理接口
*
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
*
* extern "C"
*/
extern "C" void Det_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, float* output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

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


/*
* 分割推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* output: result of pridict ,include label_map
*
* extern "C"
*/
extern "C" void Seg_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, unsigned char* output)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	std::vector<uint8_t> result_map = results[0].seg_result->label_map.data; // vector<uint8_t> -- 结果map
	// 拷贝输出结果到输出上返回 -- 将vector<uint8_t>转成unsigned char *
	memcpy(output, &result_map[0], result_map.size() * sizeof(uchar));
}


/*
* 识别推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* score: result of pridict ,include score
*
* category: result of pridict ,include category_string
*
* category_id: result of pridict ,include category_id
*
* extern "C"
*/
extern "C" void Cls_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, float* score, char* category, int* category_id)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	//cv::imwrite("./1.png", input);
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	*category_id = results[0].clas_result->category_id;
	// 拷贝输出类别结果到输出上返回 -- string --> char*
	memcpy(category, results[0].clas_result->category.c_str(), strlen(results[0].clas_result->category.c_str()));
	// 拷贝输出概率值返回
	*score = results[0].clas_result->score;
}


/*
* MaskRCNN推理接口
*
* img: input for predicting.
*
* nWidth: width of img.
*
* nHeight: height of img.
*
* nChannel: channel of img.
*
* box_output: result of pridict ,include label+score+bbox
*
* mask_output: result of pridict ,include label_map
*
* nBoxesNum: result of pridict ,include BoxesNum
*
* LabelList: result of pridict ,include LabelList
*
* extern "C"
*/
extern "C" void Mask_ModelPredict(const unsigned char* img, int nWidth, int nHeight, int nChannel, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList)
{
	// prepare data
	std::vector<cv::Mat> imgs;

	int nType = 0;
	if (nChannel == 3)
	{
		nType = CV_8UC3;
	}
	else
	{
		std::cout << "Only support 3 channel image." << std::endl;
		return;
	}

	cv::Mat input = cv::Mat::zeros(cv::Size(nWidth, nHeight), nType);
	memcpy(input.data, img, nHeight * nWidth * nChannel * sizeof(uchar));
	imgs.push_back(std::move(input));

	// predict  -- 多次点击单张推理时会出错
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);  // 在Infer处发生错误

	nBoxesNum[0] = results[0].det_result->boxes.size();  // 得到单张图片预测的bounding box数
	std::string label = "";

	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // 得到所有框的数据
	{
		// 边界框预测结果
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		box_output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // 类别的id
		// score
		box_output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // 得分
		//// box
		box_output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		box_output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // 左上、右下的顶点
		box_output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		box_output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];

		//Mask预测结果
		for (int j = 0; j < results[0].det_result->boxes[i].mask.data.size(); j++)
		{
			if (mask_output[j] == 0)
			{
				mask_output[j] = results[0].det_result->boxes[i].mask.data[j];
			}
		}

	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* 模型销毁/注销接口
*
* extern "C"
*/
extern "C" void DestructModel()
{
	delete model;
	std::cout << "destruct model success" << std::endl;
}
