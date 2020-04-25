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
import argparse

prj_path = str(Path(__file__).parent.absolute()) + '/../'
print(f"prj_path: {prj_path}")
sys.path.append(os.path.dirname(prj_path))  # to add root of prj to path for runtime

from model.model import build_model, score_model, training_xform
from src.randomsearchtb import RandomSearchTB


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


def _clear_logs(log_dir, tb_dir):
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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='s3://sagemaker-scf/default')
    parser.add_argument('--input_path', type=str, default='model/input')
    parser.add_argument('--max_trials', type=str, default=25)
    parser.add_argument('--max_epochs', type=str, default=5)
    return parser.parse_known_args()


if __name__ == "__main__":
    start = datetime.now()
    print(f"Starting hyper parameter optimizations at: {start}")

    args, unknown = _parse_args()
    print(f"output_path: {args.output_path}")
    print(f"input_path: {args.input_path}")
    print(f"max_trials: {args.max_trials}")
    print(f"max_epochs: {args.max_epochs}")
    # distributed settings
    model_dir = args.output_path+'/models/'
    log_dir = 'logs'
    tb_dir = args.output_path+'/logs'

    _clear_logs(log_dir, tb_dir)
    x_train, x_test, y_train, y_test = _load_data(args.input_path)
    x_train, y_train = training_xform(x_train, y_train)

    tf.config.threading.set_inter_op_parallelism_threads = 0
    tf.config.threading.set_intra_op_parallelism_threads = 0

    tuner = RandomSearchTB(
        hypermodel=build_model,
        objective='loss',
        max_trials=int(args.max_trials),
        executions_per_trial=1,
        directory=log_dir,
        project_name='hyp_tune')

    tuner.tb_dir = tb_dir
    print(f"Search space: { tuner.search_space_summary() }")
    early_stopping_cb = K.callbacks.EarlyStopping(monitor='loss', patience=20)
    lr_scheduler_cb = K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)

    tuner.search(x_train, y_train,
                 epochs=int(args.max_epochs),
                 validation_data=(x_test, y_test),
                 callbacks=[early_stopping_cb, lr_scheduler_cb])

    print(tuner.results_summary())
    end = datetime.now()
    runtime = end - start
    print(f"Hyper parameter optimizations ended at: {end}")
    print(f"Runtime: {runtime}")

    # save model
    try:
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.evaluate(x_test, y_test)
        tuner_id = os.environ.get('KERASTUNER_TUNER_ID')
        save_prefix = model_dir + '/'
        if tuner_id:
            save_prefix += tuner_id + '/'

        best_model.save(os.path.join(save_prefix, 'trained'))
        print('best model:')
        print(f"used hyperparams: {tuner.get_best_hyperparameters(1)[0].values}")

        # show score metrics
        score_model(best_model, x_train, x_test, y_train, y_test)
    except ValueError as error:
        # probably another slave already did, print error and exit cleanly
        print(error)
