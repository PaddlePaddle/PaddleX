#!/usr/bin/env bash

set -euo pipefail

paddlex_version="$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')"
if [ -z "${paddlex_version}" ]; then
    echo "Error: no git tag found to derive version" >&2
    exit 1
fi

for device_type in 'gpu' 'cpu'; do
    docker rmi \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex${paddlex_version%.*}-${device_type}" \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:$(git rev-parse --short HEAD)-${device_type}" \
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
done
