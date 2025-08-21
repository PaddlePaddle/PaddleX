#!/usr/bin/env bash

python scripts/assemble.py "$@"

# TODO: Better way to handle the permission problem
if [ -d output ]; then
    chown -R "${OUID}":"${OGID}" output
fi
