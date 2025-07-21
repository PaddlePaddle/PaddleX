from paddlex_hps_server import BaseTritonPythonModel, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.ts_anomaly_detection.InferRequest

    def get_result_model_type(self):
        return schemas.ts_anomaly_detection.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.csv)
        df = utils.csv_bytes_to_data_frame(file_bytes)
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        result = list(self.pipeline.predict(df))[0]

        output_csv = utils.base64_encode(utils.data_frame_to_bytes(result["anomaly"]))
        if visualize_enabled:
            output_image = utils.base64_encode(
                utils.image_to_bytes(result.img["res"].convert("RGB"))
            )
        else:
            output_image = None

        return schemas.ts_anomaly_detection.InferResult(
            csv=output_csv, image=output_image
        )
