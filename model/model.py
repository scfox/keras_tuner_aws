import tensorflow as tf
from tensorflow import keras as K
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, log_loss


def build_model(hp):
    initializer = K.initializers.lecun_normal()
    layers = K.layers
    model = K.models.Sequential()
    n_hidden = hp.Int('n_hidden', min_value=2, max_value=4)

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(35, activation=tf.nn.selu, kernel_initializer=initializer))
    model.add(layers.Dropout(rate=0.1))

    # add hidden layers based on hyperparameter
    for i in range(0, n_hidden):
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(35, activation=tf.nn.selu, kernel_initializer=initializer))
        model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(1, activation=tf.nn.selu, kernel_initializer=initializer))
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999),
                  metrics=[tf.keras.metrics.binary_accuracy])
    return model


def score_model(model, x_train, x_test, y_train, y_test):
    """Score model performance"""
    print("\nScores on training:")
    y_train_pred = model.predict_classes(x_train)
    show_metrics(y_train, y_train_pred)

    print("\nScores on test:")
    y_test_pred = model.predict_classes(x_test)
    show_metrics(y_test, y_test_pred)


def show_metrics(y, y_pred):
    print(f"log_loss: {log_loss(y, y_pred)}")
    print(f"f1: {f1_score(y, y_pred , average='macro')}")
    print(f"precision: {precision_score(y, y_pred , average='macro')}")
    print(f"recall: {recall_score(y, y_pred , average='macro')}")
    print(f"confusion matrix:")
    print(f"{confusion_matrix(y, y_pred)}")


