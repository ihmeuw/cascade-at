#!/bin/bash
#$ -P proj_dismod_at
#$ -N avas_food
#$ -q all.q
#$ -l h_rt=4:00:00
#$ -l fthread=2
#$ -l m_mem_free=16G
#$ -t 1-3580
#$ -tc 20
#$ -cwd
# 3580
source /ihme/code/adolgert/venv/cascade3.7/bin/activate
python main_success_scenario.py
if [[ "$?" -ne "0" ]]
then
  echo Dismod-AT failed to run
fi
