import tensorflow as tf
from kerastuner.tuners import RandomSearch
from tensorboard.plugins.hparams import api as hp
import os

class RandomSearchTB(RandomSearch):
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
        return

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        config = trial.hyperparameters.get_config()
        hparams = self.get_hparams(config)
        if trial.score is not None:
            score = trial.score
        else:
            # in distributed training sometimes trial.score is set to 0 incorrectly
            # print(f"loss metric: {trial.metrics.metrics['loss']._observations}")
            score = self.pull_loss_from_metrics(trial)
        log_prefix = self.log_dir+'/hparam_tuning/run-'
        tuner_id = os.environ.get('KERASTUNER_TUNER_ID')
        if tuner_id:
            log_prefix += tuner_id + '-'
        with tf.summary.create_file_writer(log_prefix+str(self.count)).as_default():
            hp.hparams(hparams)
            tf.summary.scalar(self.objective, score, step=1)
        self.count += 1

    def pull_loss_from_metrics(self, trial):
        loss = 1.0
        print("Warning: trial.score is 0.  Pulling score from loss metrics....")
        loss_metric = trial.metrics.metrics.get('loss')
        if loss_metric:
            for key in loss_metric._observations:
                o = loss_metric._observations[key]
                if len(o.value) > 0:
                    if o.value[0] < loss:
                        loss = o.value[0]
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


