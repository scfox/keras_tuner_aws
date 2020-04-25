#!/bin/bash
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="0.0.0.0"
export KERASTUNER_ORACLE_PORT="8000"
pip install -r src/requirements.txt
tensorboard --logdir=${2}/logs --port 6012 --bind_all &
python src/tune_model.py --output_path=$2 --max_trials=$3 --max_epochs=$4
