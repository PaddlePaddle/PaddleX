#!/usr/bin/env bash

paddlex_version="$(cat paddlex_version.txt)"

for device_type in 'gpu' 'cpu'; do
    version="$(cat "${device_type}_version.txt")"
    docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
    for tag in "${version}-${device_type}" "paddlex${paddlex_version}-${device_type}"; do
        docker tag "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}" "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
        docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
    done
done
