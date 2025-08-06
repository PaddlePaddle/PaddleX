from paddlex.inference.pipelines.components import IndexData
from paddlex_hps_server import schemas

from common.base_model import BaseFaceRecognitionModel


class TritonPythonModel(BaseFaceRecognitionModel):
    def get_input_model_type(self):
        return schemas.face_recognition.RemoveImagesFromIndexRequest

    def get_result_model_type(self):
        return schemas.face_recognition.RemoveImagesFromIndexResult

    def run(self, input, log_id):
        index_storage = self.context["index_storage"]
        index_data_bytes = index_storage.get(input.indexKey)
        index_data = IndexData.from_bytes(index_data_bytes)

        index_data = self.pipeline.remove_index(input.ids, index_data)

        index_data_bytes = index_data.to_bytes()
        index_storage.set(input.indexKey, index_data_bytes)

        return schemas.face_recognition.RemoveImagesFromIndexResult(
            imageCount=len(index_data.id_map)
        )
