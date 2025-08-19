#!/usr/bin/env bash

for file in ../../../paddlex/configs/pipelines/*.yaml; do
    cp "${file}" "pipelines/$(basename ${file%.yaml})/server/pipeline_config.yaml"
done
