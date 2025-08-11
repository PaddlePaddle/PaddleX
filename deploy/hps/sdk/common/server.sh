#!/usr/bin/env bash

set -e

export LANG='C.UTF-8'
export PADDLEX_HPS_LOGGING_LEVEL='INFO'

export PADDLEX_HPS_PIPELINE_CONFIG_PATH="${PADDLEX_HPS_PIPELINE_CONFIG_PATH:-$(realpath pipeline_config.yaml)}"

readonly MODEL_REPO_DIR=/paddlex/var/paddlex_model_repo

rm -rf "${MODEL_REPO_DIR}"

cp -r model_repo "${MODEL_REPO_DIR}"

find "${MODEL_REPO_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' dir_; do
    if [ -f "${dir_}/config_${PADDLEX_HPS_DEVICE_TYPE}.pbtxt" ]; then
        cp -f "${dir_}/config_${PADDLEX_HPS_DEVICE_TYPE}.pbtxt" "${dir_}/config.pbtxt"
    fi
done

if [ -d shared_mods ]; then
    export PYTHONPATH="$(realpath shared_mods):${PYTHONPATH}"
fi

exec tritonserver --model-repository="${MODEL_REPO_DIR}" --backend-config=python,shm-default-byte-size=104857600,shm-growth-byte-size=10485760 --log-info=1 --log-warning=1 --log-error=1
