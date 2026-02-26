#!/usr/bin/env bash

paddlex_version="$(cat ../../../paddlex/.version)"

for device_type in 'gpu' 'cpu'; do
    docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
    for tag in "$(git rev-parse --short HEAD)-${device_type}" "paddlex${paddlex_version%.*}-${device_type}"; do
        docker tag "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}" "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
        docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
    done
done
