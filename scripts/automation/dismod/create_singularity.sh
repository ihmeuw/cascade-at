#!/bin/sh
#$ -P proj_dismod
#$ -pe multi_slot 5
#$ -cwd
#$ -j y
#$ -o /share/temp/sgeoutput/adolgert/output.txt
#$ -N SingDismod
#$ -S /bin/bash

set -x
THREADS=5
IMAGE=dismod_at:0.0.1
./build_singularity.sh -i "dismod/${IMAGE}" -f "/share/singularity-images/dismod/${IMAGE}.img" -t "${THREADS}"

#/share/singularity-images/build_singularity.sh -i "dismod/${IMAGE}" -f "/share/singularity-images/dismod/${IMAGE}.img" -t "${THREADS}"
