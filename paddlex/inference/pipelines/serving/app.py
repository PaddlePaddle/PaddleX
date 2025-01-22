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
import json
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import aiohttp
import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from typing_extensions import ParamSpec

from ..base import BasePipeline
from .models import NoResultResponse
from .utils import call_async, generate_log_id

SERVING_CONFIG_KEY: Final[str] = "Serving"

_PipelineT = TypeVar("_PipelineT", bound=BasePipeline)
_P = ParamSpec("_P")
_R = TypeVar("_R")


class PipelineWrapper(Generic[_PipelineT]):
    def __init__(self, pipeline: _PipelineT) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._lock = asyncio.Lock()

    @property
    def pipeline(self) -> _PipelineT:
        return self._pipeline

    async def infer(self, *args: Any, **kwargs: Any) -> List[Any]:
        def _infer() -> List[Any]:
            output = list(self._pipeline(*args, **kwargs))
            return output

        return await self.call(_infer)

    async def call(
        self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        async with self._lock:
            return await call_async(func, *args, **kwargs)


class AppConfig(BaseModel):
    extra: Optional[Dict[str, Any]] = None


class AppContext(Generic[_PipelineT]):
    def __init__(self, *, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self.extra: Dict[str, Any] = {}
        self._pipeline: Optional[PipelineWrapper[_PipelineT]] = None
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def pipeline(self) -> PipelineWrapper[_PipelineT]:
        if not self._pipeline:
            raise AttributeError("`pipeline` has not been set.")
        return self._pipeline

    @pipeline.setter
    def pipeline(self, val: PipelineWrapper[_PipelineT]) -> None:
        self._pipeline = val

    @property
    def aiohttp_session(self) -> aiohttp.ClientSession:
        if not self._aiohttp_session:
            raise AttributeError("`aiohttp_session` has not been set.")
        return self._aiohttp_session

    @aiohttp_session.setter
    def aiohttp_session(self, val: aiohttp.ClientSession) -> None:
        self._aiohttp_session = val


def create_app_config(pipeline_config: Mapping[str, Any], **kwargs: Any) -> AppConfig:
    app_config = pipeline_config.get(SERVING_CONFIG_KEY, {})
    app_config.update(kwargs)
    return AppConfig.model_validate(app_config)


def create_app(
    *, pipeline: _PipelineT, app_config: AppConfig, app_aiohttp_session: bool = True
) -> Tuple[fastapi.FastAPI, AppContext[_PipelineT]]:
    @contextlib.asynccontextmanager
    async def _app_lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
        ctx.pipeline = PipelineWrapper[_PipelineT](pipeline)
        if app_aiohttp_session:
            async with aiohttp.ClientSession(
                cookie_jar=aiohttp.DummyCookieJar()
            ) as aiohttp_session:
                ctx.aiohttp_session = aiohttp_session
                yield
        else:
            yield

    app = fastapi.FastAPI(lifespan=_app_lifespan)
    ctx = AppContext[_PipelineT](config=app_config)
    app.state.context = ctx

    @app.get("/health", operation_id="checkHealth")
    async def _check_health() -> NoResultResponse:
        return NoResultResponse(
            logId=generate_log_id(), errorCode=0, errorMsg="Healthy"
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: fastapi.Request, exc: RequestValidationError
    ) -> JSONResponse:
        json_compatible_data = jsonable_encoder(
            NoResultResponse(
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
            NoResultResponse(
                logId=generate_log_id(), errorCode=exc.status_code, errorMsg=exc.detail
            )
        )
        return JSONResponse(content=json_compatible_data, status_code=exc.status_code)

    return app, ctx
