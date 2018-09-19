#!/bin/bash
set -x

# Installs the develop branch of Cascade from github into a virtual environment in
# /ihme/code/dismod_at/env
# The environment is made current if all the tests pass
# EpiViz calls a script to submit a Cascade model, which calls a script which runs dismod.
# That second script can activate the virtual environment installed here.  

usage() {
    cat <<-EOF
    This installs the develop branch of Cascade into /ihme/code/dismod_at/env.

    Usage: $0 

EOF
}

if [ $# -gt 0 ]; then
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        usage
        exit 0
    fi
fi


# Check that the user qlogin into a node, not just login to dev or prod 

which mysql_config 2>/dev/null
if [ "$?" -ne "0" ]; then
    echo Cannot find mysql_config. Are you working from cluster-dev or cluster-prod?
    echo Run this from a node.
fi
   
# Check that PYTHONPATH is not set

if [ ! -z "${PYTHONPATH}" ]; then
    echo PYTHONPATH is set. Unsetting. PYTHONPATH overrides virtual
    echo environment when using source activate.
    unset PYTHONPATH
fi


# We're going to create a virtual env containing the project code
# Check that the expected installed python containing virtualenv exists

PYTHON_MINICONDA=/ihme/code/dismod_at/pyenv/versions/current
VIRTUALENV="${PYTHON_MINICONDA}/bin/virtualenv"

if ! test -e "${VIRTUALENV}"; then
    echo Cannot find ${VIRTUALENV}
    exit 14
fi

# Download the Cascade project from github

CASCADE_HOME=/ihme/code/dismod_at
CASCADE_DEVELOP_DIR=${CASCADE_HOME}/cascade_develop

GITHUB="https://github.com/ihmeuw/cascade.git"

if [ -e "${CASCADE_DEVELOP_DIR}" ]; then
    git -C "${CASCADE_DEVELOP_DIR}" pull
else
    git clone "${GITHUB}" "${CASCADE_DEVELOP_DIR}"    
fi

# Create the env directory and allocate a new name for the env for this install

ENVDIR=${CASCADE_HOME}/env
mkdir -p "${ENVDIR}"

COUNTER=0
ENVNAME=`date +%Y%m%d`
while [ -e "${ENVDIR}/${ENVNAME}" ]
do
    COUNTER=$((COUNTER + 1))
    ENVNAME=`date +%Y%m%d`-${COUNTER}
done

ENV=${ENVDIR}/${ENVNAME}
echo $ENV

# Create the virtual environment for the project code and required packages

"${PYTHON_MINICONDA}/bin/python" -m venv "${ENV}"

source "${ENV}/bin/activate"
pip install --upgrade pip
cd "${CASCADE_DEVELOP_DIR}"
pip install .[ihme_databases,documentation,testing]


# Install the virtual environment to be the "current" if the tests pass

(cd "${CASCADE_DEVELOP_DIR}/tests" && pytest)

if [ "$?" -eq "0" ]; then
    for softlink in prod dev current
    do
        rm -f "${ENVDIR}/${softlink}"; ln -sf "${ENV}" "${ENVDIR}/${softlink}"
    done
    echo Installed ${ENV}
else
    echo Tests not all passed for ${ENV}
fi


# Validate the dismod_at binary file name and executable attribute:
# Note: we have been using two different locations 
# for the dismod_at singularity image
# /share/singularity-images/dismod/current.img
# /share/singularity-images/dismod_at/current.img

DISMOD_AT_PATH=`readlink -f /share/singularity-images/dismod/current.img`
if ! [ -f "${DISMOD_AT_PATH}" ] ; then
    echo "ERROR: the DISMOD_AT_PATH is invalid!"
    exit 1
fi
echo "The physical path of the DISMOD_AT executable is ${DISMOD_AT_PATH}."
