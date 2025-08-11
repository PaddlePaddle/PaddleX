from typing import Any, Dict, List

import numpy as np
import pycocotools.mask as mask_util
from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


def _rle(mask: np.ndarray) -> str:
    rle_res = mask_util.encode(np.asarray(mask[..., None], order="F", dtype="uint8"))[0]
    return rle_res["counts"].decode("utf-8")


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.instance_segmentation.InferRequest

    def get_result_model_type(self):
        return schemas.instance_segmentation.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(file_bytes)
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        result = list(self.pipeline.predict(image, threshold=input.threshold))[0]

        instances: List[Dict[str, Any]] = []
        for obj, mask in zip(result["boxes"], result["masks"]):
            rle_res = _rle(mask)
            mask = dict(rleResult=rle_res, size=mask.shape)
            instances.append(
                dict(
                    bbox=obj["coordinate"],
                    categoryId=obj["cls_id"],
                    categoryName=obj["label"],
                    score=obj["score"],
                    mask=mask,
                )
            )
        if visualize_enabled:
            output_image_base64 = utils.base64_encode(
                utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return schemas.instance_segmentation.InferResult(
            instances=instances, image=output_image_base64
        )
