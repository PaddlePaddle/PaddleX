#!/usr/bin/env bash

# TODO: Allow building multiple variants in a single run?

set -e

device_type='gpu'
env_type='prod'
docker_image_tag='latest'
pip_index_url=''
paddlex_version=''

while getopts 'k:dt:p:v:' opt; do
    case "${opt}" in 
        k)
            # "k" for "kind"
            if [ "${OPTARG}" != 'cpu' ] && [ "${OPTARG}" != 'gpu' ]; then
                echo "Invalid device type: ${OPTARG}" >&2
                exit 2
            fi
            device_type="${OPTARG}"
            ;;
        d) env_type='dev' ;;
        t) docker_image_tag="${OPTARG}" ;;
        p) pip_index_url="${OPTARG}" ;;
        v) paddlex_version="${OPTARG}" ;;

        *) 
            echo "Usage: $0 [-k <device-type>] [-d] [-t <docker-image-tag>] [-p <pip-index-url>] [-v <paddlex-version>]" >&2
            exit 2
            ;;
    esac
done
shift "$((OPTIND - 1))"
if [ $# -ne 0 ]; then
    echo "Positional parameters will not be used." >&2
fi

if [ -z "${paddlex_version}" ]; then
    paddlex_tag="$(git describe --tags --exact-match --match 'v[0-9]*' HEAD 2>/dev/null || true)"
    paddlex_version="${paddlex_tag#v}"
fi

args=(
  -f Dockerfile
  -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${docker_image_tag}"
  --target deployment
  --build-arg DEVICE_TYPE="${device_type}"
  --build-arg ENV_TYPE="${env_type}"
  --build-arg PADDLEX_VERSION="${paddlex_version}"
  --build-arg http_proxy="${http_proxy}"
  --build-arg https_proxy="${https_proxy}"
)

[ -n "${pip_index_url}" ] && args+=( --build-arg PIP_INDEX_URL="${pip_index_url}" )


DOCKER_BUILDKIT=1 docker build "${args[@]}" ../../..
