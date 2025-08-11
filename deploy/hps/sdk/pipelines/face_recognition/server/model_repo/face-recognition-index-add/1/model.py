from operator import attrgetter

from paddlex.inference.pipelines.components import IndexData
from paddlex_hps_server import schemas, utils

from common.base_model import BaseFaceRecognitionModel


class TritonPythonModel(BaseFaceRecognitionModel):
    def get_input_model_type(self):
        return schemas.face_recognition.AddImagesToIndexRequest

    def get_result_model_type(self):
        return schemas.face_recognition.AddImagesToIndexResult

    def run(self, input, log_id):
        file_bytes_list = [
            utils.get_raw_bytes(img)
            for img in map(attrgetter("image"), input.imageLabelPairs)
        ]
        images = [utils.image_bytes_to_array(item) for item in file_bytes_list]
        labels = [pair.label for pair in input.imageLabelPairs]

        index_storage = self.context["index_storage"]
        index_data_bytes = index_storage.get(input.indexKey)
        index_data = IndexData.from_bytes(index_data_bytes)

        index_data = self.pipeline.append_index(images, labels, index_data)

        index_data_bytes = index_data.to_bytes()
        index_storage.set(input.indexKey, index_data_bytes)

        return schemas.face_recognition.AddImagesToIndexResult(
            imageCount=len(index_data.id_map)
        )
