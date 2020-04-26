
# class to mimic hyperparameters to train one specific model with specific hyperparameters


class HypParams:
    def Int(self, name, **kwargs):
        if name == 'n_hidden':
            return 1
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