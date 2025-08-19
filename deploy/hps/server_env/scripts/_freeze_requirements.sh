#!/usr/bin/env bash

# TODO: Better way to handle the permission problem

# Should we use `pip-compile-multi`?

set -e

python -m piptools compile \
    -q \
    -o "requirements/${DEVICE_TYPE}.txt" \
    --allow-unsafe \
    --strip-extras \
    --no-emit-index-url \
    --no-emit-trusted-host \
    --extra 'base' \
    --extra 'serving' \
    requirements/app.in "requirements/${DEVICE_TYPE}.in" paddlex-hps-server/pyproject.toml ../../../setup.py
chown "${OUID}":"${OGID}" "requirements/${DEVICE_TYPE}.txt"

python -m piptools compile \
    -q \
    -c "requirements/${DEVICE_TYPE}.txt" \
    -o "requirements/${DEVICE_TYPE}_hpi.txt" \
    --allow-unsafe \
    --strip-extras \
    --no-emit-index-url \
    --no-emit-trusted-host \
    "requirements/${DEVICE_TYPE}_hpi.in"
chown "${OUID}":"${OGID}" "requirements/${DEVICE_TYPE}_hpi.txt"

python -m piptools compile \
    -q \
    -c "requirements/${DEVICE_TYPE}.txt" \
    -o "requirements/${DEVICE_TYPE}_dev.txt" \
    --allow-unsafe \
    --strip-extras \
    --no-emit-index-url \
    --no-emit-trusted-host \
    "requirements/${DEVICE_TYPE}_dev.in" 
chown "${OUID}":"${OGID}" "requirements/${DEVICE_TYPE}_dev.txt"
