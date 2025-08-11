from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

    def get_input_model_type(self):
        return schemas.pp_chatocrv4_doc.InvokeMLLMRequest

    def get_result_model_type(self):
        return schemas.pp_chatocrv4_doc.InvokeMLLMResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(file_bytes)

        mllm_predict_info = self.pipeline.mllm_pred(
            image,
            input.keyList,
            mllm_chat_bot_config=input.mllmChatBotConfig,
        )

        return schemas.pp_chatocrv4_doc.InvokeMLLMResult(
            mllmPredictInfo=mllm_predict_info
        )
