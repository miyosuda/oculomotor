#!/bin/sh

CONTAINER_APP=/opt/oculomotor

docker run -it --rm -v ${PWD}:${CONTAINER_APP} wbap/oculomotor python ${CONTAINER_APP}/application/collect.py --opt_flow_only=true $*
