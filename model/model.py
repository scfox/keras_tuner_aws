import tensorflow as tf
from tensorflow import keras as K
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, log_loss
from imblearn.over_sampling import SMOTE


def build_model(hp):
    initializer = K.initializers.lecun_normal()
    layers = K.layers
    model = K.models.Sequential()
    n_hidden = hp.Int('n_hidden', min_value=2, max_value=6)
    dropout_rate = hp.Float('dropout_rate', min_value=0.06, max_value=0.1, sampling='linear')
    init_lr = hp.Fixed('init_lr', value=.001)
    beta1 = hp.Choice('beta1', values=[0.88, 0.95])
    # ol_act = hp.Choice('ol_act', values=['selu', 'relu', 'sigmoid'])
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(35, activation=tf.nn.selu, kernel_initializer=initializer))
    model.add(layers.Dropout(rate=dropout_rate))
    # add hidden layers based on hyperparameter
    for i in range(0, n_hidden):
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(35, activation=tf.nn.selu, kernel_initializer=initializer))
        model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Nadam(lr=init_lr, beta_1=beta1, beta_2=0.999),
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


def training_xform(x_train, y_train):
    print("Applying OverSampling ----")
    print(f"Before OverSampling, counts of label '1': {sum(y_train == 1)}")
    print(f"Before OverSampling, counts of label '0': {sum(y_train == 0)}")
    sm = SMOTE(random_state=23)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    print(f"After OverSampling, counts of label '1': {sum(y_train == 1)}")
    print(f"After OverSampling, counts of label '0': {sum(y_train == 0)}")
    return x_train, y_train

