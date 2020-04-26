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

from model.model import build_model, score_model, training_xform, f1b
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


class HypParams:
    def Int(self, name, **kwargs):
        if name == 'n_hidden':
            return 2
        else:
            return 0

    def Float(self, name, **kwargs):
        if name == 'dropout_rate':
            return 0.1
        elif name == 'beta1':
            return 0.930
        elif name == 'beta2':
            return 0.9945
        else:
            return 0


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='../logs/output')
    parser.add_argument('--input_path', type=str, default='../model/input')
    parser.add_argument('--max_epochs', type=str, default=2)
    return parser.parse_known_args()


if __name__ == "__main__":
    start = datetime.now()
    print(f"Starting training of model at: {start}")

    args, unknown = _parse_args()
    print(f"output_path: {args.output_path}")
    print(f"input_path: {args.input_path}")
    print(f"max_epochs: {args.max_epochs}")
    # distributed settings
    model_dir = args.output_path+'/models'
    log_dir = '../logs'
    tb_dir = args.output_path+'/logs'

    _clear_logs(log_dir, tb_dir)
    x_train, x_test, y_train, y_test = _load_data(args.input_path)
    x_train, y_train = training_xform(x_train, y_train)

    tf.config.threading.set_inter_op_parallelism_threads = 0
    tf.config.threading.set_intra_op_parallelism_threads = 0

    save_prefix = model_dir + '/'
    checkpoint_path = os.path.join(save_prefix, 'cp')

    hp = HypParams()
    model = build_model(hp)

    early_stopping_cb = K.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.00001,
                                                  restore_best_weights=True,
                                                  verbose=1,
                                                  patience=7)
    checkpoint_cb = K.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_best_only=True,
                                                verbose=1,
                                                monitor='val_loss')
    lr_scheduler_cb = K.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    min_delta=0.0001,
                                                    factor=0.5,
                                                    patience=2,
                                                    verbose=1)
    tensorboard_cb = K.callbacks.TensorBoard(log_dir)
    model.fit(x_train, y_train,
              epochs=int(args.max_epochs),
              validation_data=(x_test, y_test),
              batch_size=8,
              callbacks=[early_stopping_cb, lr_scheduler_cb, checkpoint_cb, tensorboard_cb])

    end = datetime.now()
    runtime = end - start
    print(f"Model training ended at: {end}")
    print(f"Runtime: {runtime}")

    # show score metrics
    score_model(model, x_train, x_test, y_train, y_test)

    # load last checkpoint
    cp_model = tf.keras.models.load_model(checkpoint_path)
    print(f"\n--- Last Checkpoint model performance: ")
    score_model(cp_model, x_train, x_test, y_train, y_test)

