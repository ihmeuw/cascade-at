#!/bin/bash
set -x

# This script installs miniconda and numpy into the dismod_at space, 
# and upgrades pip, virtualenv, and pipenv

CODE=/ihme/code/dismod_at
WORKDIR="${CODE}/build"
SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
INSTALLDIR="${CODE}/pyenv/versions"

# Any existing python path can interfere with installation, as can
# other Pythons on the path.
unset PYTHONPATH
export PATH=/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin

cd "${WORKDIR}"
rm -f "${SCRIPT}"
wget "https://repo.continuum.io/miniconda/${SCRIPT}"

# The version is stored in this way in the file, so find that.
# NAME:   Miniconda3
# VER:    4.5.1
CONDANAME=`awk '/NAME:/ {print tolower($3)}' ${SCRIPT}`
CONDAVER=`awk '/VER:/ {print $3}' ${SCRIPT}`
SUBDIR="${CONDANAME}-${CONDAVER}"
CONDAHOME="${INSTALLDIR}/${SUBDIR}"

if [ -e  "${CONDAHOME}" ]; then
  echo Directory "${CONDAHOME}" exists so cannot install.
  exit 37
fi

# Run the installer in batch mode.
bash "${SCRIPT}" -b -p "${CONDAHOME}"

"${CONDAHOME}/bin/conda" install -y arrow blosc hdf5 mysql-connector-c numpy[mkl] redis
"${CONDAHOME}/bin/pip" install --upgrade pip virtualenv pipenv

# Lock it down so we can't accidentally modify it.
chmod -R a-w "${CONDAHOME}"
