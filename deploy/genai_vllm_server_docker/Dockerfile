ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.18.1-x86_64
FROM ${VLLM_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 wget \
    && rm -rf /var/lib/apt/lists/* \
    && if ! command -v python >/dev/null; then ln -s "$(command -v python3)" /usr/local/bin/python; fi

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG PADDLEX_VERSION
ARG PADDLEX_WHEEL
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=paddlex-wheel,target=/paddlex-build \
    if [ -n "${PADDLEX_WHEEL}" ] && [ -z "${PADDLEX_VERSION}" ]; then \
        python -m pip install "/paddlex-build/${PADDLEX_WHEEL}"; \
    elif [ -n "${PADDLEX_VERSION}" ] && [ -z "${PADDLEX_WHEEL}" ]; then \
        python -m pip install "paddlex${PADDLEX_VERSION}"; \
    else \
        echo "Set exactly one of PADDLEX_VERSION or PADDLEX_WHEEL." >&2; \
        exit 2; \
    fi \
    && paddlex --install genai-vllm-server

EXPOSE 8080

ENTRYPOINT []
CMD ["paddlex_genai_server", "--model_name", "PaddleOCR-VL-0.9B", "--host", "0.0.0.0", "--port", "8080", "--backend", "vllm"]
