#include <string>
#include <vector>

#include "model_deploy/common/include/paddle_deploy.h"

// Global model pointer
PaddleDeploy::Model* model;

/*
* Model initialization / registration API
* 
* model_type: det,seg,clas,paddlex
* 
* model_filename: Model file path
* 
* params_filename: Parameter file path
* 
* cfg_file: Configuration file path
* 
* use_gpu: Whether to use GPU
* 
* gpu_id: Specify GPU x
* 
* paddlex_model_type: When Model_Type is paddlx, the type of actual Paddlex model returned - det, seg, clas
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
	if (strcmp(model_type, "paddlex") == 0) // If it is a PADDLEX model, return the specifically supported model type: det, seg, clas
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
* Detection inference API
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
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	// nBoxesNum[0] = results.size();  // results.size() is returning batch_size
	nBoxesNum[0] = results[0].det_result->boxes.size();  // Get the predicted Bounding Box number of a single image
	std::string label = "";
	//std::cout << "res: " << results[num] << std::endl;
	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // Get the data for all the boxes
	{
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // Category ID
		// score
		output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // Score
		//// box
		output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // Upper left and lower right vertices
		output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];
	}
	memcpy(LabelList, label.c_str(), strlen(label.c_str()));
}


/*
* Segmented inference 
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
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	std::vector<uint8_t> result_map = results[0].seg_result->label_map.data; // vector<uint8_t> -- Result Map
	// Copy output result to the output back -- from vector<uint8_t> to unsigned char *
	memcpy(output, &result_map[0], result_map.size() * sizeof(uchar));
}


/*
* Recognition inference API
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
	imgs.push_back(std::move(input));

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	*category_id = results[0].clas_result->category_id;
	// Copy output category result to output -- string --> char* 
	memcpy(category, results[0].clas_result->category.c_str(), strlen(results[0].clas_result->category.c_str()));
	// Copy output probability value
	*score = results[0].clas_result->score;
}	


/*
* MaskRCNN Reasoning 
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

	// predict
	std::vector<PaddleDeploy::Result> results;
	model->Predict(imgs, &results, 1);

	nBoxesNum[0] = results[0].det_result->boxes.size();  // Get the predicted Bounding Box number of a single image
	std::string label = "";

	for (int i = 0; i < results[0].det_result->boxes.size(); i++)  // Get the data for all the boxes
	{
		// prediction results
		label = label + results[0].det_result->boxes[i].category + " ";
		// labelindex
		box_output[i * 6 + 0] = results[0].det_result->boxes[i].category_id; // Category ID
		// score
		box_output[i * 6 + 1] = results[0].det_result->boxes[i].score;  // Score
		//// box
		box_output[i * 6 + 2] = results[0].det_result->boxes[i].coordinate[0]; // x1, y1, x2, y2
		box_output[i * 6 + 3] = results[0].det_result->boxes[i].coordinate[1]; // Upper left and lower right vertices
		box_output[i * 6 + 4] = results[0].det_result->boxes[i].coordinate[2];
		box_output[i * 6 + 5] = results[0].det_result->boxes[i].coordinate[3];
		
		// Mask prediction results
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
* Model destruction API
* 
* extern "C" 
*/
extern "C" void DestructModel()
{
	delete model;
	std::cout << "destruct model success" << std::endl;
}
