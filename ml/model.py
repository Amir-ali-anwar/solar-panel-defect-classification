"""Model architecture: MobileNetV2 transfer learning with baked-in augmentation.

Transfer learning is used instead of a from-scratch CNN because the dataset is
small (~870 images across 6 classes) -- a custom CNN trained from scratch on
that little data overfits quickly, while a frozen ImageNet backbone plus a
light classification head generalizes far better and trains in a fraction of
the time.
"""
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        ],
        name="augmentation",
    )


def apply_preprocessing(dataset, training, img_size):
    augmentation = build_augmentation() if training else None

    def _prep(images, labels):
        images = tf.cast(images, tf.float32)
        if augmentation is not None:
            images = augmentation(images, training=True)
        images = preprocess_input(images)
        return images, labels

    return dataset.map(_prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def build_model(num_classes, img_size):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def unfreeze_for_fine_tuning(base_model, fine_tune_at_layer):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at_layer]:
        layer.trainable = False
