"""Central configuration shared by the notebook, the API and the retraining pipeline."""
import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MANIFEST_PATH = DATA_DIR / "split_manifest.csv"
INSIGHTS_PATH = DATA_DIR / "insights.json"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "mlops.db"))

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "coffee_leaf_model.h5"             # full Keras model, trained in the notebook
BACKBONE_PATH = MODELS_DIR / "feature_extractor.tflite"      # its exported MobileNetV2 part (used by the API)
HEAD_PATH = MODELS_DIR / "classifier_head.npz"               # its exported classifier head (used by the API)
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"                # stored features of every dataset image
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_cnn.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
ARCHIVE_DIR = MODELS_DIR / "archive"

IMG_SIZE = 128
SEED = 42
CLASS_NAMES = ["Cercospora", "Healthy", "Leaf_rust", "Miner", "Phoma"]

# Retraining starts automatically once RETRAIN_THRESHOLD new images are waiting (if AUTO_RETRAIN is on).
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "30"))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", "5e-4"))
# Images further than this from their 5 nearest training images (cosine distance between features)
# are not coffee-leaf photos: no diagnosis is given. 0.47 accepts all 688 test leaves.
LEAF_DISTANCE_THRESHOLD = float(os.getenv("LEAF_DISTANCE_THRESHOLD", "0.47"))
LEAF_NEIGHBOURS = 5

# A retrained model is promoted only if test accuracy and macro-F1 drop by no more than this.
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", "0.02"))
