from src.randomsearchtb import RandomSearchTB
from tensorflow import keras as K


def build_model(hp):
    model = K.models.Sequential()
    return model


class TestRandomSearchTB:

    def test_instantiation(self):
        # setup
        # test
        rs = RandomSearchTB(hypermodel=build_model, objective='binary_accuracy', max_trials=1)
        # verify
        assert rs is not None

    def test_hparam_from_config_space_item(self):
        # setup
        rs = RandomSearchTB(hypermodel=build_model, objective='binary_accuracy', max_trials=1)
        c = {
            'class_name': 'int',
            'config': {
                'name': 'n_hidden',
                'default': None,
                'min_value': 2,
                'max_value': 4,
                'step': 1,
                'sampling': None,
            }
        }
        # test
        h = rs.hparam_from_config_space_item(c_item=c)
        # verify
        assert type(h).__name__ == 'HParam'
        assert h.name == 'n_hidden'

    def test_value_for_config_space_item(self):
        # setup
        rs = RandomSearchTB(hypermodel=build_model, objective='binary_accuracy', max_trials=1)
        c_item = {
            'class_name': 'int',
            'config': {
                'name': 'n_hidden',
                'default': None,
                'min_value': 2,
                'max_value': 4,
                'step': 1,
                'sampling': None,
            }
        }
        config = {
            'space': [c_item],
            'values': {'n_hidden': 3},
        }
        # test
        val = rs.value_for_config_space_item(c_item, config)
        # verify
        assert val == 3

    def test_get_hparams(self):
        # setup
        rs = RandomSearchTB(hypermodel=build_model, objective='binary_accuracy', max_trials=1)
        c_item = {
            'class_name': 'int',
            'config': {
                'name': 'n_hidden',
                'default': None,
                'min_value': 2,
                'max_value': 4,
                'step': 1,
                'sampling': None,
            }
        }
        config = {
            'space': [c_item],
            'values': {'n_hidden': 3},
        }
        # test
        h = rs.get_hparams(config)
        # verify
        assert type(h).__name__ == 'dict'
        assert len(h) == 1
