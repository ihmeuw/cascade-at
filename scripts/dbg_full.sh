#!/bin/bash
# This script is for debugging. The run_cascade.sh will call it
# so that we can see whether run_cascade.sh is working.

OUTDIR="/share/temp/sgeoutput/$USER/errors"
mkdir -p "${OUTDIR}"
OUTFILE="${OUTDIR}/cascade_debug.log"
env | sort > "${OUTFILE}"
exec run_full.sh $1 1
