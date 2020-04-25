#!/bin/bash
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
export KERASTUNER_TUNER_ID=`curl http://169.254.169.254/latest/meta-data/local-ipv4`
export KERASTUNER_ORACLE_IP="$1"
export KERASTUNER_ORACLE_PORT="8000"
pip install -r src/requirements.txt
while :
do
  python src/tune_model.py --output_path=$2 --max_trials=$3 --max_epochs=$4
  echo "restarting..."
done
