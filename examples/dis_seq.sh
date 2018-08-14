#!/bin/bash
# For installation
#   1.  Install Docker
#   2.  Be on VPN
#   3.  Get a copy of Cascade by cloning https://github.com/ihmeuw/cascade
#   4.  Install Cascade by activating a Python virtual environment and running
#         pip install .
#
# Then run this with "bash dis_seq.sh".

# Turns on verbose Bash scripting.
set -x

# This tells Docker to make the current directory available to Dismod in
# the Docker container as the /app directory, so ./fit.db becomes /app/fit.db.
CURDIR=`pwd`
MOUNT="type=bind,source=${CURDIR},target=/app"
# Location of Dismod source code within the Docker
DISMOD_SRC=/usr/src/app/dismod_at
# Location of Dismod C++ executable within the Docker
DISMOD=/home/root/prefix/dismod_at/bin/dismod_at
# Location of Dismod Python executable within the Docker.
DISPY=/home/root/prefix/dismod_at/bin/dismodat.py

# Let's see if we have the latest version. Go to Docker registry and check.
docker pull reg.ihme.washington.edu/dismod/dismod_at
# This one runs in the dismod_at home directory. -it gets you interactive.
DISPREFIX="docker run -it --mount ${MOUNT} reg.ihme.washington.edu/dismod/dismod_at"


# Runs the simple application that's in the cascade repository.
dmcsv2db
FIT=/app/fit.db
${DISPREFIX} bash -c "${DISMOD} ${FIT} set option quasi_fixed false"
${DISPREFIX} bash -c "${DISMOD} ${FIT} set option ode_step_size 1"
${DISPREFIX} bash -c "${DISMOD} ${FIT} init"
${DISPREFIX} bash -c "${DISMOD} ${FIT} fit fixed"
${DISPREFIX} bash -c "${DISMOD} ${FIT} predict fit_var"
