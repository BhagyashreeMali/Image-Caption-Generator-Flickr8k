from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

CAPTIONS_FILE = RAW_DIR / "Flickr8k.token.txt"
IMAGES_DIR = RAW_DIR / "Flickr8k_images"
VOCAB_PATH = PROCESSED_DIR / "vocab.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
MAX_LEN = 30
MIN_WORD_FREQ = 5   