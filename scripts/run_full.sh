#!/bin/bash
# "${ENVIRONMENT_NAME}" is passed in and should be dev or prod
MVID=$1
if [ "x${MVID}" = "x" ]; then
    echo Must supply model version ID.
    exit 4
fi

if [ $# -gt 1 ]; then
    DEBUG=$2
else
    DEBUG=0
fi

if test -r ~/EPIAT_DIR
then
    # Custom conda install + environment, for development
    EPI_DIR="$(cat ~/EPIAT_DIR)"
else
    EPI_DIR="/ihme/code/dismod_at/env/${SGE_CLUSTER_NAME}"
fi
if ! test -d ${EPI_DIR}
then
    echo Cannot find directory ${EPI_DIR}
    exit 7
fi

if [ "${DEBUG}" = 1 ]; then
    RUNNER="dmcascade -v"
else
    RUNNER="dmcascade"
fi
DB_DIR=$(mktemp -d "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")
source "${EPI_DIR}/bin/activate"
echo "${RUNNER}" "$DB_DIR/model.db" --mvid "${MVID}"
"${RUNNER}" "$DB_DIR/model.db" --mvid "${MVID}"
