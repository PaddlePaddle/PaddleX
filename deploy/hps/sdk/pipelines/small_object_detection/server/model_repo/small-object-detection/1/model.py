from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.small_object_detection.InferRequest

    def get_result_model_type(self):
        return schemas.small_object_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(file_bytes)
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        result = list(self.pipeline.predict(image, threshold=input.threshold))[0]

        objects: List[Dict[str, Any]] = []
        for obj in result["boxes"]:
            objects.append(
                dict(
                    bbox=obj["coordinate"],
                    categoryId=obj["cls_id"],
                    categoryName=obj["label"],
                    score=obj["score"],
                )
            )
        if visualize_enabled:
            output_image_base64 = utils.base64_encode(
                utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return schemas.small_object_detection.InferResult(
            detectedObjects=objects, image=output_image_base64
        )
