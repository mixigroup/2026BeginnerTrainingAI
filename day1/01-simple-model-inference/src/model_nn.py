"""Neural Network model definition and training for Iris classification."""

from datetime import datetime

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


def build_model(
    input_dim: int = 4,
    hidden_units: list[int] | None = None,
    dropout_rate: float = 0.2,
    num_classes: int = 3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build and compile a fully-connected Neural Network for classification.

    Architecture:
        Input(input_dim) -> Dense(hidden_units[0], ReLU)
                         -> Dense(hidden_units[1], ReLU)
                         -> Dense(hidden_units[2], ReLU)
                         -> Dropout(dropout_rate)
                         -> Dense(num_classes, Softmax)

    Args:
        input_dim: Number of input features (default 4 for Iris)
        hidden_units: List of hidden layer sizes (default [1000, 500, 300])
        dropout_rate: Dropout probability for regularization (default 0.2)
        num_classes: Number of output classes (default 3 for Iris)
        learning_rate: Adam optimizer learning rate (default 0.001)

    Returns:
        Compiled Keras Sequential model
    """
    if hidden_units is None:
        hidden_units = [1000, 500, 300]

    layers = [Dense(hidden_units[0], activation="relu", input_shape=(input_dim,))]
    for units in hidden_units[1:]:
        layers.append(Dense(units, activation="relu"))
    layers.append(Dropout(dropout_rate))
    layers.append(Dense(num_classes, activation="softmax"))

    model = Sequential(layers)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: keras.Model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    batch_size: int = 100,
    epochs: int = 100,
    log_dir: str | None = None,
) -> keras.callbacks.History:
    """Train the model with optional TensorBoard logging.

    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels (one-hot encoded)
        X_valid: Validation features
        y_valid: Validation labels (one-hot encoded)
        batch_size: Mini-batch size (default 100)
        epochs: Number of training epochs (default 100)
        log_dir: Directory for TensorBoard logs. If None, uses 'logs/scalars/<timestamp>'

    Returns:
        Keras History object containing loss/accuracy per epoch
    """
    if log_dir is None:
        log_dir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir)]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )
    return history
