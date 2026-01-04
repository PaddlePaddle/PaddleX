# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict, List

from .....utils.deps import function_requires_deps, is_dep_available
from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import AIStudioResultResponse
from ...schemas.multilingual_speech_recognition import (
    INFER_ENDPOINT,
    InferRequest,
    InferResult,
)
from .._app import create_app, primary_operation

if is_dep_available("fastapi"):
    from fastapi import FastAPI, HTTPException


@function_requires_deps("fastapi")
def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> "FastAPI":
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> AIStudioResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        file_bytes = await serving_utils.get_raw_bytes_async(
            request.audio, aiohttp_session
        )
        ext = serving_utils.infer_file_ext(request.audio)
        if ext is None:
            raise HTTPException(
                status_code=422, detail="File extension cannot be inferred"
            )
        audio_path = await serving_utils.call_async(
            serving_utils.write_to_temp_file,
            file_bytes,
            suffix=ext,
        )

        try:
            result = (await pipeline.infer(audio_path))[0]
        finally:
            await serving_utils.call_async(os.unlink, audio_path)

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

        return AIStudioResultResponse[InferResult](
            logId=serving_utils.generate_log_id(),
            result=InferResult(
                text=result["result"]["text"],
                segments=segments,
                language=result["result"]["language"],
            ),
        )

    return app
