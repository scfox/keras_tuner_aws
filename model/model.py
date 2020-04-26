import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import backend as KBk
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, log_loss, accuracy_score
from imblearn.over_sampling import SMOTE


def build_model(hp):
    initializer = K.initializers.lecun_normal()
    layers = K.layers
    model = K.models.Sequential()
    n_hidden = hp.Int('n_hidden', min_value=1, max_value=2)
    dropout_rate = hp.Float('dropout_rate', min_value=0.06, max_value=0.1, sampling='linear')
    init_lr = 0.001  # hp.Fixed('init_lr', value=.001)
    beta1 = hp.Float('beta1', min_value=0.88, max_value=0.95, sampling='linear')
    beta2 = hp.Float('beta2', min_value=0.98, max_value=0.999, sampling='linear')
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
    model.compile(loss=K.losses.binary_crossentropy,  # 'binary_crossentropy', K.metrics.BinaryAccuracy
                  optimizer=tf.keras.optimizers.Nadam(lr=init_lr, beta_1=beta1, beta_2=beta2),
                  metrics=[tf.keras.metrics.binary_accuracy])
    return model


def score_model(model, x_train, x_test, y_train, y_test):
    """Score model performance"""
    print("\nScores on training:")
    y_train_pred = [float(p[0]) for p in model.predict_classes(x_train)]
    y_train_proba = [float(p[0]) for p in model.predict(x_train)]
    show_metrics(y_train, y_train_pred, y_train_proba)
    # q = model.predict_proba(x_train)
    print("\nScores on test:")
    y_test_vals = [i[0] for i in y_test]
    y_test_pred = [float(p[0]) for p in model.predict_classes(x_test)]
    y_train_proba = [float(p[0]) for p in model.predict(x_test)]
    show_metrics(y_test_vals, y_test_pred, y_train_proba)


def show_metrics(y, y_pred, y_proba):
    print(f"loss: {K.losses.binary_crossentropy(y, y_proba)}")
    print(f"log_loss: {log_loss(y, y_pred)}")
    print(f"f1: {f1_score(y, y_pred , average='macro')}")
    print(f"precision: {precision_score(y, y_pred , average='macro')}")
    print(f"recall: {recall_score(y, y_pred , average='macro')}")
    print(f"accuracy_score: {accuracy_score(y, y_pred )}")
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


def f1b(y_true, y_pred):
    true_positives = KBk.sum(KBk.round(KBk.clip(y_true * y_pred, 0, 1)))
    possible_positives = KBk.sum(KBk.round(KBk.clip(y_true, 0, 1)))
    predicted_positives = KBk.sum(KBk.round(KBk.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + KBk.epsilon())
    recall = true_positives / (possible_positives + KBk.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+KBk.epsilon())
    return f1_val
