#!/bin/sh

CONTAINER_APP=/opt/oculomotor

docker run -d -it --rm -v ${PWD}:${CONTAINER_APP} wbap/oculomotor python ${CONTAINER_APP}/application/collect.py $*

