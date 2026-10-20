FROM python:3.10

RUN apt-get update \
    && apt-get install -y libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/cuda-toolkit/cuda-debian12-13.0.0-580.65.06-1-amd64.deb \
    && dpkg -i cuda-debian12-13.0.0-580.65.06-1-amd64.deb \
    && cp /var/cuda-repo-debian12-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get -y install cuda-toolkit-13-0 \
    && rm -f cuda-debian12-13.0.0-580.65.06-1-amd64.deb \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG PADDLEX_VERSION=">=3.6"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "paddlex${PADDLEX_VERSION}"

RUN --mount=type=cache,target=/root/.cache/pip \
   paddlex --install genai-vllm-server

EXPOSE 8080

CMD ["paddlex_genai_server", "--model_name", "PaddleOCR-VL-1.5-0.9B", "--host", "0.0.0.0", "--port", "8080", "--backend", "vllm"]
