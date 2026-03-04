#!/usr/bin/env bash

paddlex_version="$(cat ../../../paddlex/.version)"

for device_type in 'gpu' 'cpu'; do
    docker rmi \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex${paddlex_version%.*}-${device_type}" \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:$(git rev-parse --short HEAD)-${device_type}" \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
done
