#!/bin/bash
source /home/ec2-user/anaconda3/bin/activate tensorflow2_p36
export KERASTUNER_TUNER_ID=`curl http://169.254.169.254/latest/meta-data/local-ipv4`
export KERASTUNER_ORACLE_IP="$1"
export KERASTUNER_ORACLE_PORT="8000"
pip install -r src/requirements.txt
while :
do
  python src/tune_model.py --output_path=$2 --max_trials=$3 --max_epochs=$4
  if [ $? -eq 0 ]
  then
    echo "tune_model completed."
    aws s3 cp scripts/complete.txt ${2}/complete/${KERASTUNER_TUNER_ID}
    break
  else
    echo "tune model failed.  restarting..."
    aws s3 cp /var/log/cloud-init-output.log ${2}/error/${KERASTUNER_TUNER_ID}.log
    echo "restarting..."
  fi
done
