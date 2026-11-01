"""Shared preprocessing and prediction adapter for the tabular 1D CNN."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config.settings import CANONICAL_22_FEATURES, RANDOM_SEED

CATEGORICAL_FEATURES = ["proto", "flgs", "state"]
NUMERIC_FEATURES = [
    name for name in CANONICAL_22_FEATURES if name not in CATEGORICAL_FEATURES
]


def make_preprocessors():
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=1000,
                    output_distribution="normal",
                    subsample=100_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int32,
    )
    label_encoder = LabelEncoder()
    return numeric_transformer, categorical_encoder, label_encoder


def fit_transform_features(X, numeric_transformer, categorical_encoder):
    numeric = numeric_transformer.fit_transform(X[NUMERIC_FEATURES]).astype(
        np.float32, copy=False
    )
    categories = categorical_encoder.fit_transform(X[CATEGORICAL_FEATURES])
    return _as_model_inputs(numeric, categories)


def transform_features(X, numeric_transformer, categorical_encoder):
    numeric = numeric_transformer.transform(X[NUMERIC_FEATURES]).astype(
        np.float32, copy=False
    )
    categories = categorical_encoder.transform(X[CATEGORICAL_FEATURES])
    return _as_model_inputs(numeric, categories)


def _as_model_inputs(numeric, categories):
    inputs = {"numeric_sequence": numeric[:, :, np.newaxis]}
    for index, feature in enumerate(CATEGORICAL_FEATURES):
        # Reserve index zero for an unseen category; known IDs start at one.
        inputs[f"cat_{feature}"] = np.maximum(categories[:, index], -1).astype(
            np.int32
        )[:, np.newaxis] + 1
    return inputs


def build_cnn(cardinalities, n_classes):
    """Build a 1D CNN over ordered numeric features with categorical embeddings."""
    import tensorflow as tf

    numeric_input = tf.keras.Input(
        shape=(len(NUMERIC_FEATURES), 1), name="numeric_sequence"
    )
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, padding="same", activation="relu"
    )(numeric_input)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    inputs = {"numeric_sequence": numeric_input}
    representations = [x]
    for feature, cardinality in zip(CATEGORICAL_FEATURES, cardinalities):
        category_input = tf.keras.Input(shape=(1,), dtype="int32", name=f"cat_{feature}")
        embedding = tf.keras.layers.Embedding(
            input_dim=int(cardinality) + 1,
            output_dim=min(16, max(4, int(np.ceil(np.sqrt(cardinality))))),
            name=f"embedding_{feature}",
        )(category_input)
        representations.append(tf.keras.layers.Flatten()(embedding))
        inputs[f"cat_{feature}"] = category_input

    x = tf.keras.layers.Concatenate()(representations)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    output = tf.keras.layers.Dense(n_classes, activation="softmax", name="category")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNCategoryPredictor:
    """Expose a saved Keras CNN through the predict_proba interface."""

    def __init__(
        self, network, numeric_transformer, categorical_encoder, label_encoder
    ):
        self.network = network
        self.numeric_transformer = numeric_transformer
        self.categorical_encoder = categorical_encoder
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model_inputs = transform_features(
            X, self.numeric_transformer, self.categorical_encoder
        )
        return self.network.predict(model_inputs, verbose=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        class_ids = np.argmax(probabilities, axis=1)
        return self.label_encoder.inverse_transform(class_ids)
