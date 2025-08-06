from typing import Any, Dict, List

from paddlex.inference.pipelines.components import IndexData
from paddlex_hps_server import schemas, utils

from common.base_model import BaseFaceRecognitionModel


class TritonPythonModel(BaseFaceRecognitionModel):
    def get_input_model_type(self):
        return schemas.face_recognition.InferRequest

    def get_result_model_type(self):
        return schemas.face_recognition.InferResult

    def run(self, input, log_id):
        image_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(image_bytes)

        if input.indexKey is not None:
            index_storage = self.context["index_storage"]
            index_data_bytes = index_storage.get(input.indexKey)
            index_data = IndexData.from_bytes(index_data_bytes)
        else:
            index_data = None
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        result = list(
            self.pipeline(
                image,
                index=index_data,
                det_threshold=input.detThreshold,
                rec_threshold=input.recThreshold,
                hamming_radius=input.hammingRadius,
                topk=input.topk,
            )
        )[0]

        objs: List[Dict[str, Any]] = []
        for obj in result["boxes"]:
            rec_results: List[Dict[str, Any]] = []
            if obj["rec_scores"] is not None:
                for label, score in zip(obj["labels"], obj["rec_scores"]):
                    rec_results.append(
                        dict(
                            label=label,
                            score=score,
                        )
                    )
            objs.append(
                dict(
                    bbox=obj["coordinate"],
                    recResults=rec_results,
                    score=obj["det_score"],
                )
            )
        if visualize_enabled:
            output_image_base64 = utils.base64_encode(
                utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return schemas.face_recognition.InferResult(
            faces=objs, image=output_image_base64
        )
