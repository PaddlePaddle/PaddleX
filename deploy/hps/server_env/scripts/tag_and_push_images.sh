#!/usr/bin/env bash

set -euo pipefail

paddlex_tag="$(git describe --tags --abbrev=0 2>/dev/null || true)"
if [ -z "${paddlex_tag}" ]; then
    echo "Error: no git tag found to derive version" >&2
    exit 1
fi

if ! [[ "${paddlex_tag}" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "Error: expected a release tag in format vX.Y.Z, got '${paddlex_tag}'" >&2
    exit 1
fi

paddlex_minor_version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"

for device_type in 'gpu' 'cpu'; do
    docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
    for tag in "$(git rev-parse --short HEAD)-${device_type}" "paddlex${paddlex_minor_version}-${device_type}"; do
        docker tag "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}" "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
        docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
    done
done
