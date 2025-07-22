import os
from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, protocol, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.video_detection.InferRequest

    def get_result_model_type(self):
        return schemas.video_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.video)
        ext = utils.infer_file_ext(input.video)
        if ext is None:
            return protocol.create_aistudio_output_without_result(
                422,
                "File extension cannot be inferred",
                log_id=log_id,
            )
        video_path = utils.write_to_temp_file(
            file_bytes,
            suffix=ext,
        )

        try:
            result = list(
                self.pipeline(
                    video_path,
                    nms_thresh=input.nmsThresh,
                    score_thresh=input.scoreThresh,
                )
            )[0]
        finally:
            os.unlink(video_path)

        frames: List[Dict[str, Any]] = []
        for i, item in enumerate(result["result"]):
            objs: List[Dict[str, Any]] = []
            for obj in item:
                objs.append(
                    dict(
                        bbox=obj[0],
                        categoryName=obj[2],
                        score=obj[1],
                    )
                )
            frames.append(
                dict(
                    index=i,
                    detectedObjects=objs,
                )
            )

        return schemas.video_detection.InferResult(frames=frames)
