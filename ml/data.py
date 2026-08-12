"""Dataset discovery, stratified splitting, and tf.data pipeline construction."""
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def discover_samples(data_dir: Path):
    """Walk class subdirectories and return (filepaths, labels, class_names)."""
    data_dir = Path(data_dir)
    class_names = sorted(
        p.name for p in data_dir.iterdir() if p.is_dir()
    )
    if not class_names:
        raise FileNotFoundError(f"No class subdirectories found under {data_dir}")

    filepaths, labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        # rglob (not iterdir) because some classes (e.g. Bird-drop/New/) nest
        # extra images in subdirectories one level below the class folder.
        for f in sorted(class_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                filepaths.append(str(f))
                labels.append(label_idx)

    if not filepaths:
        raise FileNotFoundError(f"No image files found under {data_dir}")

    return filepaths, np.array(labels), class_names


def stratified_split(filepaths, labels, train_split, val_split, test_split, seed):
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("train/val/test splits must sum to 1.0")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filepaths,
        labels,
        train_size=train_split,
        random_state=seed,
        stratify=labels,
    )

    remaining = val_split + test_split
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_split / remaining,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def oversample_to_balance(paths, labels, seed):
    """Duplicate minority-class samples (train split only) up to the size of
    the largest class. Relies on the heavy augmentation already applied per
    epoch (flip/rotation/zoom/contrast/translation/brightness) so duplicated
    files aren't seen identically each time -- this changes how *often* the
    model sees minority classes, which class-weighting the loss alone does
    not (that only reweights the gradient, not the sampling frequency)."""
    rng = np.random.default_rng(seed)
    paths = np.asarray(paths)
    labels = np.asarray(labels)

    counts = np.bincount(labels)
    target = counts.max()

    extra_paths, extra_labels = [], []
    for class_idx, count in enumerate(counts):
        if count == 0 or count >= target:
            continue
        class_indices = np.where(labels == class_idx)[0]
        needed = target - count
        chosen = rng.choice(class_indices, size=needed, replace=True)
        extra_paths.append(paths[chosen])
        extra_labels.append(labels[chosen])

    if extra_paths:
        paths = np.concatenate([paths, *extra_paths])
        labels = np.concatenate([labels, *extra_labels])

    shuffle_idx = rng.permutation(len(paths))
    return paths[shuffle_idx].tolist(), labels[shuffle_idx]


def _decode_and_resize(path_tensor, img_size):
    """Decode + resize via PIL, using JPEG draft mode to downscale during
    decode. Source photos here go up to ~6240x4160 (~78MB decoded as
    uint8); decoding full-resolution then resizing can exceed available
    RAM on memory-constrained machines. draft() lets libjpeg decode at a
    fraction of that resolution up front instead."""
    path = path_tensor.numpy().decode("utf-8")
    with Image.open(path) as img:
        img.draft("RGB", img_size)
        img = img.convert("RGB").resize(img_size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def _load_image(path, label, img_size):
    image = tf.py_function(lambda p: _decode_and_resize(p, img_size), [path], tf.float32)
    image.set_shape([*img_size, 3])
    return image, label


def build_dataset(paths, labels, img_size, batch_size, shuffle, seed):
    # Source photos go up to ~6240x4160 (~78MB decoded as uint8). Capping
    # decode parallelism and prefetch depth keeps peak memory bounded on
    # memory-constrained machines -- AUTOTUNE here can spike well past
    # available RAM before the resize step shrinks each image down.
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _load_image(p, l, img_size),
        num_parallel_calls=2,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def load_splits(data_dir, img_size, batch_size, train_split, val_split, test_split, seed):
    filepaths, labels, class_names = discover_samples(data_dir)
    splits = stratified_split(filepaths, labels, train_split, val_split, test_split, seed)

    original_train_size = len(splits["train"][0])
    balanced_train_paths, balanced_train_labels = oversample_to_balance(
        splits["train"][0], splits["train"][1], seed
    )
    splits["train"] = (balanced_train_paths, balanced_train_labels)

    datasets = {
        name: build_dataset(
            paths, lbls, img_size, batch_size, shuffle=(name == "train"), seed=seed
        )
        for name, (paths, lbls) in splits.items()
    }

    logger.info(
        "Dataset sizes -> train: %d (oversampled from %d), val: %d, test: %d (classes: %s)",
        len(splits["train"][0]),
        original_train_size,
        len(splits["val"][0]),
        len(splits["test"][0]),
        class_names,
    )

    class_weights = compute_class_weights(splits["train"][1], len(class_names))
    return datasets, class_names, class_weights, splits
