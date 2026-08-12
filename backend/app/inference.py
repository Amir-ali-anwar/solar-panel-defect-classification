"""Model loading and prediction logic, isolated behind ModelService so tests
can substitute a stub without needing the real trained artifact."""
import io
import json
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    pass


class ModelService:
    def __init__(self, model_path, class_names_path):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names: list[str] = []
        self.img_size: tuple[int, int] = (224, 224)

    def load(self):
        import tensorflow as tf  # deferred: keep TF off the import path for tests that stub this out

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run `python -m ml.train` first."
            )
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found at {self.class_names_path}.")

        logger.info("Loading model from %s", self.model_path)
        self.model = tf.keras.models.load_model(self.model_path)
        # Derive the expected input resolution from the model itself so it
        # always matches whatever size ml/train.py was run with.
        _, height, width, _ = self.model.input_shape
        self.img_size = (width, height)  # PIL Image.resize takes (width, height)
        with open(self.class_names_path) as f:
            self.class_names = json.load(f)
        logger.info("Model loaded. Classes: %s. Input size: %s", self.class_names, self.img_size)

    @property
    def is_loaded(self):
        return self.model is not None

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise InvalidImageError("Uploaded file is not a valid image") from exc

    def _build_tta_batch(self, image: Image.Image) -> np.ndarray:
        """Build a small batch of augmented views for test-time augmentation.

        Averaging predictions over a few views (flip, slight zoom) instead of
        a single pass is a well-known, no-retrain way to reduce variance in
        the model's predictions -- useful here since the training set is
        small enough that single-view confidence can be noisy.
        """
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        base = image.resize(self.img_size)
        flipped = base.transpose(Image.FLIP_LEFT_RIGHT)

        width, height = image.size
        crop_w, crop_h = int(width * 0.9), int(height * 0.9)
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        zoomed = image.crop((left, top, left + crop_w, top + crop_h)).resize(self.img_size)

        views = [base, flipped, zoomed]
        arrays = [preprocess_input(np.array(v, dtype=np.float32)) for v in views]
        return np.stack(arrays, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        image = self._load_image(image_bytes)
        batch = self._build_tta_batch(image)
        probs = self.model.predict(batch, verbose=0).mean(axis=0)

        probabilities = {name: float(p) for name, p in zip(self.class_names, probs)}
        best_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.class_names[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": probabilities,
        }
