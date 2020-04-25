import tensorflow as tf
from tensorflow import keras as K
from datetime import datetime
from kerastuner.tuners import RandomSearch
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from pathlib import Path
import sys
import boto3

prj_path = str(Path(__file__).parent.absolute()) + '/../'
print(f"prj_path: {prj_path}")
sys.path.append(os.path.dirname(prj_path))  # to add root of prj to path for runtime

from model.model import build_model, score_model, training_xform
from src.randomsearchtb import RandomSearchTB

# distributed settings
model_dir = 's3://sagemaker-scf/catchjoe/models/'
log_dir = 'logs'
tb_dir = 's3://sagemaker-scf/catchjoe/logs'

# local settings
# model_dir = 'model/'
# log_dir = 'logs'
# tb_dir = 'logs/tb_dir'


def _load_data(base_dir):
    print(f"Reading from :{base_dir}")
    if base_dir:
        x = pd.read_csv(os.path.join(base_dir, 'x.csv')).to_numpy()
        y = pd.read_csv(os.path.join(base_dir, 'y.csv')).to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
        return x_train, x_test, y_train, y_test
    else:
        print('Error: No input path specified.')
        return None, None, None, None


def _clear_logs():
    # delete logs from any previous run
    shutil.rmtree(log_dir, ignore_errors=True, onerror=None)
    if tb_dir[:3] == 's3:':
        # Logging to  s3
        s3 = boto3.resource('s3')
        bucket_name = tb_dir.split('/')[2]
        key = tb_dir.split(bucket_name)[1][1:]
        s3.Object(bucket_name, key).delete()
    else:
        # Logging to file
        shutil.rmtree(tb_dir, ignore_errors=True, onerror=None)


if __name__ == "__main__":
    start = datetime.now()
    print(f"Starting hyper parameter optimizations at: {start}")
    _clear_logs()
    x_train, x_test, y_train, y_test = _load_data('model/input')
    x_train, y_train = training_xform(x_train, y_train)

    tf.config.threading.set_inter_op_parallelism_threads = 0
    tf.config.threading.set_intra_op_parallelism_threads = 0
    tf.config.threading

    tuner = RandomSearchTB(
        hypermodel=build_model,
        objective='loss',
        max_trials=25,
        executions_per_trial=1,
        directory=log_dir,
        project_name='catchjoe')

    tuner.tb_dir = tb_dir
    print(f"Search space: { tuner.search_space_summary() }")
    early_stopping_cb = K.callbacks.EarlyStopping(monitor='loss', patience=20)
    lr_scheduler_cb = K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)

    tuner.search(x_train, y_train,
                 epochs=20,
                 validation_data=(x_test, y_test),
                 callbacks=[early_stopping_cb, lr_scheduler_cb])

    print(tuner.results_summary())
    end = datetime.now()
    runtime = end - start
    print(f"Hyper parameter optimizations ended at: {end}")
    print(f"Runtime: {runtime}")

    # save model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.evaluate(x_test, y_test)
    tuner_id = os.environ.get('KERASTUNER_TUNER_ID')
    save_prefix = model_dir
    if tuner_id:
        save_prefix += '-' + tuner_id + '/'

    best_model.save(os.path.join(save_prefix, 'trained'))
    print('best model:')
    print(f"used hyperparams: {tuner.get_best_hyperparameters(1)[0].values}")

    # show score metrics
    score_model(best_model, x_train, x_test, y_train, y_test)
