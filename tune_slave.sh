export KERASTUNER_TUNER_ID="tuner0"
export KERASTUNER_ORACLE_IP="$1"
export KERASTUNER_ORACLE_PORT="8000"
python src/tune_model.py