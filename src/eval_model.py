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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/trained')
    parser.add_argument('--input_path', type=str, default='../model/input')
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    print(f"Evaluating model: {args.model_path}")

    x_train, x_test, y_train, y_test = _load_data(args.input_path)
    x_train, y_train = training_xform(x_train, y_train)

    # load model
    model = tf.keras.models.load_model(args.model_path)

    # show score metrics
    score_model(model, x_train, x_test, y_train, y_test)
