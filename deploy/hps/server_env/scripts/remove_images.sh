#!/usr/bin/env bash

paddlex_version="$(cat ../../paddlex/.version)"

for device_type in 'gpu' 'cpu'; do
    version="$(cat "${device_type}_version.txt")"
    docker rmi \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex${paddlex_version%.*}-${device_type}" \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${version}-${device_type}" \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
done
