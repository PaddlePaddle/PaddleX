import os
from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.m_3d_bev_detection.InferRequest

    def get_result_model_type(self):
        return schemas.m_3d_bev_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.tar)
        tar_path = utils.write_to_temp_file(
            file_bytes,
            suffix=".tar",
        )

        try:
            result = list(
                self.pipeline(
                    tar_path,
                )
            )[0]
        finally:
            os.unlink(tar_path)

        objects: List[Dict[str, Any]] = []
        for box, label, score in zip(
            result["boxes_3d"], result["labels_3d"], result["scores_3d"]
        ):
            objects.append(
                dict(
                    bbox=box,
                    categoryId=label,
                    score=score,
                )
            )

        return schemas.m_3d_bev_detection.InferResult(detectedObjects=objects)
