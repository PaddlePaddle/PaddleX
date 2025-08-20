import os
from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, protocol, schemas, utils


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.multilingual_speech_recognition.InferRequest

    def get_result_model_type(self):
        return schemas.multilingual_speech_recognition.InferResult

    def run(self, input, log_id):
        file_bytes = utils.get_raw_bytes(input.audio)
        ext = utils.infer_file_ext(input.audio)
        if ext is None:
            return protocol.create_aistudio_output_without_result(
                422,
                "File extension cannot be inferred",
                log_id=log_id,
            )
        audio_path = utils.write_to_temp_file(
            file_bytes,
            suffix=ext,
        )

        try:
            result = list(self.pipeline(audio_path))[0]
        finally:
            os.unlink(audio_path)

        segments: List[Dict[str, Any]] = []
        for item in result["result"]["segments"]:
            segment = dict(
                id=item["id"],
                seek=item["seek"],
                start=item["start"],
                end=item["end"],
                text=item["text"],
                tokens=item["tokens"],
                temperature=item["temperature"],
                avgLogProb=item["avg_logprob"],
                compressionRatio=item["compression_ratio"],
                noSpeechProb=item["no_speech_prob"],
            )
            segments.append(segment)

        return schemas.multilingual_speech_recognition.InferResult(
            text=result["result"]["text"],
            segments=segments,
            language=result["result"]["language"],
        )
