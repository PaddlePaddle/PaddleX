# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import asyncio
import contextlib
import functools
import json
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Type

import aiohttp
import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from ..pipelines.base import BasePipeline
from .models import Response
from .utils import config_logger, generate_log_id


class PipelineWrapper(object):
    def __init__(self, pipeline: BasePipeline) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._lock = asyncio.Lock()

    async def infer(self, data: Any) -> Any:
        def _infer(pipeline: BasePipeline, input_: Any) -> Any:
            output = list(pipeline(input_))
            if len(output) != 1:
                raise RuntimeError("Expected exactly one item from the generator")
            return output[0]

        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(_infer, data)
            )


class AppConfig(BaseModel):
    pipeline_init_kwargs: Dict[str, Any]
    config_logger: bool = False
    debug: bool = False
    extra: Optional[Dict[str, Any]] = None


class AppContext(object):
    def __init__(self, *, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._pipeline: Optional[PipelineWrapper] = None
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def pipeline(self) -> PipelineWrapper:
        if not self._pipeline:
            raise AttributeError("`pipeline` has not been set.")
        return self._pipeline

    @pipeline.setter
    def pipeline(self, val: PipelineWrapper) -> None:
        self._pipeline = val

    @property
    def aiohttp_session(self) -> aiohttp.ClientSession:
        if not self._aiohttp_session:
            raise AttributeError("`aiohttp_session` has not been set.")
        return self.aiohttp_session

    @aiohttp_session.setter
    def aiohttp_session(self, val: aiohttp.ClientSession) -> None:
        self._aiohttp_session = val


def create_app(
    *,
    pipeline_cls: Type[BasePipeline],
    app_config: AppConfig,
    app_aiohttp_session: bool = True
) -> Tuple[fastapi.FastAPI, AppContext]:
    @contextlib.asynccontextmanager
    async def _app_lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
        ctx.pipeline = pipeline_cls(**ctx.config.pipeline_init_kwargs)
        if app_aiohttp_session:
            ctx.aiohttp_session = aiohttp.ClientSession(
                cookie_jar=aiohttp.DummyCookieJar()
            )
        if ctx.config.config_logger:
            config_logger(debug=ctx.config.debug)
        yield
        if app_aiohttp_session:
            await ctx.aiohttp_session.close()

    app = fastapi.FastAPI(lifespan=_app_lifespan)
    ctx = AppContext(config=app_config)

    @app.get("/health", operation_id="checkHealth")
    async def _check_health() -> Response:
        return Response(logId=generate_log_id(), errorCode=0, errorMsg="Healthy")

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: fastapi.Request, exc: RequestValidationError
    ) -> JSONResponse:
        json_compatible_data = jsonable_encoder(
            Response(
                logId=generate_log_id(),
                errorCode=422,
                errorMsg=json.dumps(exc.errors()),
            )
        )
        return JSONResponse(content=json_compatible_data, status_code=422)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        request: fastapi.Request, exc: HTTPException
    ) -> JSONResponse:
        json_compatible_data = jsonable_encoder(
            Response(
                logId=generate_log_id(), errorCode=exc.status_code, errorMsg=exc.detail
            )
        )
        return JSONResponse(content=json_compatible_data, status_code=exc.status_code)

    return app, ctx
