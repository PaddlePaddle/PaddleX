#!/usr/bin/env bash

python scripts/assemble.py "$@"

# TODO: Better way to handle the permission problem
chown -R "${OUID}":"${OGID}" output
