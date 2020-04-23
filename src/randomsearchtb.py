import tensorflow as tf
from kerastuner.tuners import RandomSearch
from tensorboard.plugins.hparams import api as hp


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
        return

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        config = trial.hyperparameters.get_config()
        hparams = self.get_hparams(config)
        with tf.summary.create_file_writer(self.log_dir+'/hparam_tuning/run-'+str(self.count)).as_default():
            hp.hparams(hparams)
            tf.summary.scalar('binary_accuracy', trial.score, step=1)
        self.count += 1

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
        hparam = hp.HParam(name=name, domain=hp.Discrete([2, 3, 4]))
        return hparam

    def value_for_config_space_item(self, c_item, config):
        name = c_item['config']['name']
        return config['values'][name]


