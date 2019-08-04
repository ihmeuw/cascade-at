#!/bin/bash
# "${ENVIRONMENT_NAME}" is passed in and should be dev or prod
MVID=$1
if [[ "x${MVID}" = "x" ]]; then
    echo Must supply model version ID.
    exit 4
fi

if [[ $# -gt 1 ]]; then
    DEBUG=$2
else
    DEBUG=0
fi

if test -r ~/EPIAT_DIR
then
    # Custom conda install + environment, for development
    EPI_DIR="$(cat ~/EPIAT_DIR)"
else
    if [[ "${SGE_ENV}" =~ ^prod ]]; then
	    CLUSTER="prod"
    else
	    CLUSTER="dev"
    fi
    EPI_DIR="/ihme/code/dismod_at/env/${CLUSTER}"
fi
if ! test -d "${EPI_DIR}"
then
    echo Cannot find directory ${EPI_DIR}
    exit 7
fi

if [[ "${DEBUG}" = 1 ]]; then
    RUNNER="dismodel -v"
else
    RUNNER="dismodel"
fi
DB_DIR="/ihme/epi/at_cascade/prod/${MVID}"
mkdir -p "${DB_DIR}"
source "${EPI_DIR}/bin/activate"
echo "${RUNNER}" --mvid "${MVID}" --grid-engine
"${RUNNER}" --mvid "${MVID}" --grid-engine
