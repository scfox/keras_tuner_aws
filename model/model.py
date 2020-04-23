import tensorflow as tf
from tensorflow import keras as K


class CustomModel(K.models.Sequential):
    """Encapsulates model and hyperparameters"""

    def __init__(self, hp):
        """constructor"""
        super().__init__()
        self.hp = hp
        self.num_layers = 1
        self._build_model()
        return

    def _build_model(self):
        initializer = tf.keras.initializers.lecun_normal()
        layers = tf.keras.layers

        self.add(layers.BatchNormalization())
        self.add(layers.Dense(35, activation=tf.nn.selu, kernel_initializer=initializer))
        self.add(layers.Dropout(rate=0.1))

        # add hidden layers based on hyperparameter
        for i in range(0, self.num_layers):
            self.add(layers.BatchNormalization())
            self.add(layers.Dense(35, activation=tf.nn.selu, kernel_initializer=initializer))
            self.add(layers.Dropout(rate=0.1))

        self.add(layers.Dense(1, activation=tf.nn.selu, kernel_initializer=initializer))
