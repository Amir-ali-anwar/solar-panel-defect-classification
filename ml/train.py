"""Train the solar panel condition classifier and export a production artifact.

Usage:
    python -m ml.train
    python -m ml.train --epochs 10 --fine-tune-epochs 5 --batch-size 16
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from ml import config
from ml.data import load_splits
from ml.model import apply_preprocessing, build_model, unfreeze_for_fine_tuning

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=config.INITIAL_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--skip-fine-tune", action="store_true")
    return parser.parse_args()


def plot_history(histories, output_path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (test set)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, class_names, class_weights, splits = load_splits(
        data_dir=args.data_dir,
        img_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=args.seed,
    )
    num_classes = len(class_names)
    logger.info("Class weights: %s", class_weights)

    train_ds = apply_preprocessing(datasets["train"], training=True, img_size=config.IMG_SIZE)
    val_ds = apply_preprocessing(datasets["val"], training=False, img_size=config.IMG_SIZE)
    test_ds = apply_preprocessing(datasets["test"], training=False, img_size=config.IMG_SIZE)

    steps_per_epoch = -(-len(splits["train"][1]) // args.batch_size)  # ceil division

    model, base_model = build_model(num_classes, config.IMG_SIZE)
    phase1_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.INITIAL_LEARNING_RATE,
        decay_steps=steps_per_epoch * args.epochs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(phase1_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    checkpoint_path = args.output_dir / config.MODEL_FILENAME
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    logger.info("Phase 1: training classification head (frozen backbone)")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    histories = [history1]

    if not args.skip_fine_tune:
        logger.info("Phase 2: fine-tuning top backbone layers")
        unfreeze_for_fine_tuning(base_model, config.FINE_TUNE_AT_LAYER)
        phase2_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.FINE_TUNE_LEARNING_RATE,
            decay_steps=steps_per_epoch * args.fine_tune_epochs,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(phase2_schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )
        histories.append(history2)

    model.save(checkpoint_path)
    logger.info("Saved model to %s", checkpoint_path)

    with open(args.output_dir / config.CLASS_NAMES_FILENAME, "w") as f:
        json.dump(class_names, f, indent=2)

    logger.info("Evaluating on held-out test set")
    test_labels = splits["test"][1]
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        test_labels, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    test_loss, test_accuracy = model.evaluate(test_ds)
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_accuracy, test_loss)

    with open(args.output_dir / config.METRICS_FILENAME, "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "classification_report": report,
                "img_size": config.IMG_SIZE,
                "class_names": class_names,
            },
            f,
            indent=2,
        )

    plot_history(histories, args.output_dir / config.HISTORY_PLOT_FILENAME)
    cm = confusion_matrix(test_labels, y_pred)
    plot_confusion_matrix(cm, class_names, args.output_dir / config.CONFUSION_MATRIX_FILENAME)

    logger.info("Training artifacts written to %s", args.output_dir)


if __name__ == "__main__":
    main()
