#!/bin/bash
# EpiViz calls this script in order to launch DismodAT jobs.
# It must return quickly.
#
# $0 <username> <model version ID> (<debug option>)
#
# There are debugging options that add together to be the debug option.
# 1 == Instead of sudo, run qsub as this user.
# 2 == Instead of running qsub, run echo.
# 4 == Instead of qsub of the run_full.sh, qsub dbg_full.sh
#
# Logs are in /ihme/epi/at_cascade/logs.
USER_NAME=$1
MVID=$2

if [ "x${USER_NAME}" = "x" ]; then
    echo Must supply user name and model version ID.
    exit 3
fi

if [ "x${MVID}" = "x" ]; then
    echo Must supply user name and model version ID.
    exit 4
fi

if [ $# -gt 2 ]; then
    # The third argument is a debug bitfield. We shift the bits
    # and then check which are enabled.
    DBG_SUDO=$(( $3>>0 & 1 ))
    DBG_QSUB=$(( $3>>1 & 1 ))
    DBG_SCRIPT=$(( $3>>2 & 1 ))
    if [ "$3" != 0 ]; then
      set -x
    fi
else
    DBG_SUDO=0
    DBG_QSUB=0
    DBG_SCRIPT=0
fi

LOG_DIR="/ihme/epi/at_cascade/logs"
if [ ! -e "${LOG_DIR}" ]; then
    echo Cannot find ${LOG_DIR}
    df /ihme/epi
    exit 5
fi
# This log file holds real errors and should be kept.
LOG_FILE="${LOG_DIR}/qsub_log.${MVID}.$$"

# The environment name identifies whether this is the dev or prod version
# of EpiViz. There are two separate interfaces. The dev EpiViz chooses to
# use the dev cluster, but that isn't a necessary conjunction, and the cluster
# can be checked with a different environment variable.
EPI_ENV="${ENVIRONMENT_NAME:-prod}"
if [ "${DBG_QSUB}" = 1 ]; then
    QSUB="echo"
else
    QSUB="qsub"
fi
if [ "${DBG_SCRIPT}" = 1 ]; then
    CALLSCRIPT="$(dirname $0)/dbg_full.sh"
else
    CALLSCRIPT="$(dirname $0)/run_full.sh"
fi
ROOTEDSCRIPT=`readlink -f "${CALLSCRIPT}"`

QSUB_CMD="qsub -N dmat_${MVID} -P proj_dismod -terse -o ${LOG_FILE} -j y -b y /bin/bash --noprofile --norc ${ROOTEDSCRIPT} ${MVID} ${EPI_ENV} > ${LOG_FILE} 2>&1"
if [ "${DBG_SUDO}" = 1 ]; then
    sh -c ". /etc/profile.d/sge.sh;${QSUB_CMD}"
else
    sudo -n -u "${USER_NAME}" sh -c ". /etc/profile.d/sge.sh;${QSUB_CMD}"
fi

if [ $? != 0 ]; then
    echo ${USER_NAME} >> "${LOG_FILE}"
    echo ${MVID} >> "${LOG_FILE}"
    if [ "${DBG_SUDO}" = 0 ]; then
        sudo -n chown "${USER_NAME}:IHME-users" "${LOG_FILE}"
    fi
fi
