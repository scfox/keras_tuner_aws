import tensorflow as tf
from kerastuner.tuners import RandomSearch, BayesianOptimization
from tensorboard.plugins.hparams import api as hp
import os
import pickle


class RandomSearchTB(BayesianOptimization):
    def __init__(self, **kwargs):
        """constructor"""
        super().__init__(**kwargs)
        self.count = 0
        self.hparams_def = None
        self.log_dir = '../logs'
        dir_param = kwargs.get('directory')
        if dir_param:
            self.log_dir = dir_param
        self.objective = 'val_loss'
        if kwargs.get('objective'):
            self.objective = kwargs.get('objective')
        self.trial_score = 999
        self.tb_dir = 'tb'
        return

    def run_trial(self, trial, *args, **kwargs):
        # add batch size as hyper param
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 64, 256, step=32)
        super().run_trial(trial, *args, **kwargs)

    def on_epoch_end(self, trial, model, epoch, logs=None):
        print(f"on_epoch_end trial.score: {trial.score}")
        super().on_epoch_end(trial, model, epoch, logs)
        if logs:
            if 'loss' in logs:
                self.trial_score = logs['loss']

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        if trial is None:
            print("on_trial_end: trial is None")
        else:
            print("on_trial_end: trial has a value")
        if trial.score is not None:
            score = trial.score
        else:
            # in distributed training sometimes trial.score is set to 0 incorrectly
            # set from value pulled from on_epoch_end
            score = self.trial_score
        # print(f"trial.metrics: { pickle.dumps(trial.metrics) }")
        config = trial.hyperparameters.get_config()
        hparams = self.get_hparams(config)
        # score = self.pull_loss_from_metrics(trial)
        log_prefix = self.tb_dir+'/hparam_tuning/run-'
        tuner_id = os.environ.get('KERASTUNER_TUNER_ID')
        if tuner_id:
            log_prefix += tuner_id + '-'
        with tf.summary.create_file_writer(log_prefix+str(self.count)).as_default():
            hp.hparams(hparams)
            tf.summary.scalar(self.objective, score, step=1)
        self.count += 1

    def pull_loss_from_metrics(self, trial):
        loss = 1.0
        print("Warning: trial.score is None.  Pulling score from loss metrics....")
        for m in trial.metrics.metrics:
            print(f"Metric: {m}")
        loss_metric = trial.metrics.metrics.get('loss')
        if loss_metric:
            print("loss_metric defined")
            for key in loss_metric._observations:
                o = loss_metric._observations[key]
                print(f"processing {key} observation")
                if len(o.value) > 0:
                    print(f"at least one value")
                    if o.value[0] < loss:
                        print(f"o.value[0]: {o.value[0]}")
                        loss = o.value[0]
        else:
            print("loss_metric not defined")
        return loss

    def get_hparams(self, config):
        if self.hparams_def is None:
            self.define_hparams(config)
        hparams = {self.hparam_from_config_space_item(c) : self.value_for_config_space_item(c, config)
                   for c in config['space']}
        return hparams

    def define_hparams(self, config):
        self.hparams_def = {}
        hparams = [self.hparam_from_config_space_item(c) for c in config['space']]
        metrics = [hp.Metric('loss', display_name='loss')]
        with tf.summary.create_file_writer(self.log_dir + '/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=hparams,
                metrics=metrics,
            )
        return self.hparams_def

    def hparam_from_config_space_item(self, c_item):
        name = c_item['config']['name']
        p_type = c_item['class_name']
        cfg = c_item['config']
        # parm = hp.Discrete()
        hparam = None
        if p_type == 'Int':
            hparam = hp.HParam(name=name, domain=hp.IntInterval(cfg['min_value'], cfg['max_value']))
        elif p_type == 'Float':
            hparam = hp.HParam(name=name, domain=hp.RealInterval(cfg['min_value'], cfg['max_value']))
        elif p_type == 'Fixed':
            hparam = hp.HParam(name=name, domain=hp.Discrete([cfg['value']]))
        elif p_type == 'Choice':
            hparam = hp.HParam(name=name, domain=hp.Discrete(cfg['values']))
        return hparam

    def value_for_config_space_item(self, c_item, config):
        name = c_item['config']['name']
        return config['values'][name]


