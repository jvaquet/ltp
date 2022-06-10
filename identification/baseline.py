import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def train_and_evaluate_baseline_model(X_train, y_train, X_test, y_test, num_labels=2):
    encoder = LabelBinarizer()
    y_train_bin = encoder.fit_transform(y_train)
    if num_labels == 2:
        y_train_bin = np.hstack((y_train_bin, 1 - y_train_bin))

    lm = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        lm, num_labels=num_labels, from_pt=True
    )

    tokens_train = tokenizer(X_train, padding=True, return_tensors="np").data

    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

    model.fit(
        tokens_train,
        y_train_bin,
        verbose=1,
        epochs=5,
        batch_size=16,
        callbacks=[callback],
        validation_split=0.1,
    )

    tokens_test = tokenizer(X_test, padding=True, return_tensors="np").data
    y_pred = model.predict(tokens_test)
    y_pred = [np.argmax(pred) for pred in y_pred[0]]

    print("Baseline Classification Results:")
    print(classification_report(y_test, y_pred))
