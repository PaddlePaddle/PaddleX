import uuid
from operator import attrgetter

from paddlex_hps_server import schemas, utils

from common.base_model import BaseFaceRecognitionModel


def _generate_index_key():
    return str(uuid.uuid4())


class TritonPythonModel(BaseFaceRecognitionModel):
    def get_input_model_type(self):
        return schemas.face_recognition.BuildIndexRequest

    def get_result_model_type(self):
        return schemas.face_recognition.BuildIndexResult

    def run(self, input, log_id):
        file_bytes_list = [
            utils.get_raw_bytes(img)
            for img in map(attrgetter("image"), input.imageLabelPairs)
        ]
        images = [utils.image_bytes_to_array(item) for item in file_bytes_list]
        labels = [pair.label for pair in input.imageLabelPairs]

        index_data = self.pipeline.build_index(
            images,
            labels,
            index_type="Flat",
            metric_type="IP",
        )

        index_storage = self.context["index_storage"]
        index_key = _generate_index_key()
        index_data_bytes = index_data.to_bytes()
        index_storage.set(index_key, index_data_bytes)

        return schemas.face_recognition.BuildIndexResult(
            indexKey=index_key, imageCount=len(index_data.id_map)
        )
