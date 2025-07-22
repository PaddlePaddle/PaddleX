from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.human_keypoint_detection.InferRequest

    def get_result_model_type(self):
        return schemas.human_keypoint_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(file_bytes)
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        result = list(
            self.pipeline.predict(
                image,
                det_threshold=input.detThreshold,
            )
        )[0]

        persons: List[Dict[str, Any]] = []
        for obj in result["boxes"]:
            persons.append(
                dict(
                    bbox=obj["coordinate"],
                    kpts=obj["keypoints"].tolist(),
                    detScore=obj["det_score"],
                    kptScore=obj["kpt_score"],
                )
            )
        if visualize_enabled:
            output_image_base64 = utils.base64_encode(
                utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return schemas.human_keypoint_detection.InferResult(
            persons=persons, image=output_image_base64
        )
