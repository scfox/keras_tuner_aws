#!/bin/bash
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
export KERASTUNER_TUNER_ID="tuner0"
export KERASTUNER_ORACLE_IP="$1"
export KERASTUNER_ORACLE_PORT="8000"
cd keras_tuner_aws
pip install -r src/requirements.txt
sleep 10
python src/tune_model.py --output_path=$2 --max_trials=$3 --max_epochs=$4
