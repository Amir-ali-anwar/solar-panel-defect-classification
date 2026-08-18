"""Central configuration for the training pipeline."""
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent

DATA_DIR = PROJECT_ROOT / "nootebooks" / "Data"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
SEED = 42

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 25
FINE_TUNE_AT_LAYER = 0  # unfreeze the entire backbone (was 100 -- top layers only)
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 5e-6  # lower than before since more layers are now trainable
EARLY_STOPPING_PATIENCE = 5

MODEL_FILENAME = "solar_panel_classifier.keras"
CLASS_NAMES_FILENAME = "class_names.json"
METRICS_FILENAME = "metrics.json"
HISTORY_PLOT_FILENAME = "training_history.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
