from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.anomaly_detection.InferRequest

    def get_result_model_type(self):
        return schemas.anomaly_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.image)
        image = utils.image_bytes_to_array(file_bytes)

        result = list(self.pipeline.predict(image))[0]

        pred = result["pred"][0].tolist()
        size = [len(pred), len(pred[0])]
        label_map = [item for sublist in pred for item in sublist]
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        if visualize_enabled:
            output_image_base64 = utils.base64_encode(
                utils.image_to_bytes(result.img["res"].convert("RGB"))
            )
        else:
            output_image_base64 = None

        return schemas.anomaly_detection.InferResult(
            labelMap=label_map, size=size, image=output_image_base64
        )
