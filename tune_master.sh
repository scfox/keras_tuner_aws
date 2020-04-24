#!/bin/bash
export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="0.0.0.0"
export KERASTUNER_ORACLE_PORT="8000"
while :
do
  python src/tune_model.py
  echo "restarting..."
done