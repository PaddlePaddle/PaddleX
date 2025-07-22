ARG DEVICE_TYPE=gpu
ARG ENV_TYPE=prod
ARG PIP_INDEX_URL=https://pypi.org/simple

# GPU base

FROM nvcr.io/nvidia/cuda@sha256:03681bbd11fea044ff3e3a5d65e190a6f935af30c56e03e740b27e4563f61e0f AS gpu-base

# CPU base
# ubuntu:20.04
FROM ubuntu:20.04@sha256:8feb4d8ca5354def3d8fce243717141ce31e2c428701f6682bd2fafe15388214 AS cpu-base

# Base
FROM ${DEVICE_TYPE}-base AS base

ARG PIP_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /paddlex

RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-venv \
    && python3.10 -m venv /paddlex/py310 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/paddlex/py310/bin:${PATH}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Requirement collection
FROM base AS rc
RUN python -m pip install pip-tools==7.4.1

FROM base AS gpu-build1
ARG DEVICE_TYPE=gpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /paddlex/libs /paddlex/tensorrt

RUN --mount=type=bind,source=./cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz,target=/tmp/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz \
    tar -xf /tmp/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz -C /paddlex \
    && mv /paddlex/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/* /paddlex/libs/

RUN --mount=type=bind,source=./TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz,target=/tmp/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    tar -xf /tmp/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C /paddlex \
    && mv /paddlex/TensorRT-8.6.1.6 /paddlex/tensorrt \
    && rm -rf /paddlex/tensorrt/data /paddlex/tensorrt/doc /paddlex/tensorrt/python /paddlex/tensorrt/samples

RUN wget -P /paddlex "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/deps/tritonserver/tritonserver-2.15.0-${DEVICE_TYPE}.zip" \
    && unzip "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}.zip" -d /paddlex \
    && mv "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}" /paddlex/tritonserver \
    && rm "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}.zip"

COPY ./paddlex-hps-server /paddlex/paddlex_hps_libs/paddlex-hps-server
RUN python -m pip wheel -w /paddlex/paddlex_hps_wheels --no-deps /paddlex/paddlex_hps_libs/paddlex-hps-server

# Build Stage for CPU
FROM base AS cpu-build1
ARG DEVICE_TYPE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /paddlex/libs

RUN wget -P /paddlex "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/deps/tritonserver/tritonserver-2.15.0-${DEVICE_TYPE}.zip" \
    && unzip "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}.zip" -d /paddlex \
    && mv "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}" /paddlex/tritonserver \
    && rm "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}.zip"

COPY ./paddlex-hps-server /paddlex/paddlex_hps_libs/paddlex-hps-server
RUN python -m pip wheel -w /paddlex/paddlex_hps_wheels --no-deps /paddlex/paddlex_hps_libs/paddlex-hps-server


FROM ${DEVICE_TYPE}-build1 AS build2
ARG DEVICE_TYPE
ARG ENV_TYPE

RUN --mount=type=bind,source=requirements/${DEVICE_TYPE}.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=requirements/${DEVICE_TYPE}_hpi.txt,target=/tmp/hpi_requirements.txt \
    --mount=type=bind,source=requirements/${DEVICE_TYPE}_dev.txt,target=/tmp/dev_requirements.txt \
    python -m pip install --requirement /tmp/requirements.txt --requirement /tmp/hpi_requirements.txt \
    && if [ "${ENV_TYPE}" = 'dev' ]; then \
        python -m pip install --requirement /tmp/dev_requirements.txt; \
    fi

RUN python -m pip install --no-deps /paddlex/paddlex_hps_wheels/*.whl

# GPU Deployment
FROM base AS gpu-deployment

RUN apt-get update \
    && apt-get install -y --no-install-recommends datacenter-gpu-manager \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build2 --chown=paddlex:paddlex /paddlex/tensorrt /paddlex/tensorrt
ENV LD_LIBRARY_PATH="/paddlex/tensorrt/lib:${LD_LIBRARY_PATH}"

RUN mkdir /paddlex/libs
RUN ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11 /paddlex/libs/libcublas.so \
    && ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcufft.so.10 /paddlex/libs/libcufft.so \
    && ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcurand.so.10 /paddlex/libs/libcurand.so \
    && ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcusolver.so.11 /paddlex/libs/libcusolver.so \
    && ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcusparse.so.11 /paddlex/libs/libcusparse.so

# CPU Deployment
FROM base AS cpu-deployment

# Deployment
FROM ${DEVICE_TYPE}-deployment AS deployment


RUN groupadd -g 1000 paddlex \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddlex

RUN apt-get update \
    && apt-get install -y --no-install-recommends libre2-5 libssl1.1 libb64-0d libnuma1 libarchive13 python3.10-dev libgl1-mesa-glx libglib2.0-0 libgomp1 ccache binutils build-essential texlive texlive-latex-base texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build2 --chown=paddlex:paddlex /paddlex/libs /paddlex/libs
ENV LD_LIBRARY_PATH="/paddlex/libs:${LD_LIBRARY_PATH}"

COPY --from=build2 --chown=paddlex:paddlex /paddlex/tritonserver /opt/tritonserver
ENV PATH="/opt/tritonserver/bin:${PATH}"

COPY --from=build2 --chown=paddlex:paddlex /paddlex/py310 /paddlex/py310

RUN mkdir /paddlex/var \
    && chown paddlex:paddlex /paddlex/var

USER paddlex

USER root

CMD ["/bin/bash"]
