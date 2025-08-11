from paddlex_hps_server import BaseTritonPythonModel, schemas


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

    def get_input_model_type(self):
        return schemas.pp_chatocrv3_doc.BuildVectorStoreRequest

    def get_result_model_type(self):
        return schemas.pp_chatocrv3_doc.BuildVectorStoreResult

    def run(self, input, log_id):
        vector_info = self.pipeline.build_vector(
            input.visualInfo,
            min_characters=input.minCharacters,
            block_size=input.blockSize,
            flag_save_bytes_vector=True,
            retriever_config=input.retrieverConfig,
        )

        return schemas.pp_chatocrv3_doc.BuildVectorStoreResult(vectorInfo=vector_info)
