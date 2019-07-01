#!/bin/bash
#$ -P proj_dismod_at
#$ -N avas_food
#$ -o /ihme/temp/sgeoutput/$USER/dismod
#$ -j y
#$ -q all.q
#$ -l h_rt=4:00:00
#$ -l fthread=1
#$ -l m_mem_free=16G
#$ -t 1944-3580
##$ -tc 32
#$ -cwd
# 3580
source /ihme/code/adolgert/venv/cascade3.7/bin/activate
python main_success_scenario.py
if [[ "$?" -ne "0" ]]
then
  echo Dismod-AT failed to run
fi
