import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = os.path.join(os.getcwd(), "data")
MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP", "PS", "Other Plastic", "Not Plastic Waste"]
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 35


def build_model():
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = tf.keras.layers.Dense(512, activation="relu")(base_model.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    train_path = os.path.join(DATASET_DIR, "train")
    val_path = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            "Training dataset not found. Please create the structure: data/train/<class> and data/val/<class>."
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.12,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
    )

    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
    )

    model = build_model()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "plastic_classifier.keras"),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            ),
        ],
    )

    model.save(os.path.join(MODEL_DIR, "plastic_classifier.keras"))
    print("Training complete. Model saved to saved_models/plastic_classifier.keras")
    print("Best validation accuracy:", max(history.history["val_accuracy"]))


if __name__ == "__main__":
    main()
