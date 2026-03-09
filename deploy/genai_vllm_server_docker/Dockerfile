FROM python:3.10

RUN apt-get update \
    && apt-get install -y libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch==2.8.0

ARG PADDLEX_VERSION=">=3.3.6,<3.4"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddlex${PADDLEX_VERSION}"

ARG BUILD_FOR_SM120=false
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_FOR_SM120}" = 'true' ]; then \
        python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3%2Bcu128torch2.8-cp310-cp310-linux_x86_64.whl; \
    else \
        python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl; \
    fi \
    && python -m pip install transformers==4.57.6 \
    && paddlex --install genai-vllm-server

EXPOSE 8080

CMD ["paddlex_genai_server", "--model_name", "PaddleOCR-VL-0.9B", "--host", "0.0.0.0", "--port", "8080", "--backend", "vllm"]
