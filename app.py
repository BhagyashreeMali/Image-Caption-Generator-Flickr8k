from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io

from src.model import DecoderRNN
from src.preprocessing import Vocabulary, load_captions
from src.config import *

app = FastAPI(title="Image Caption Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Building vocabulary...")
captions = load_captions(CAPTIONS_FILE)
vocab = Vocabulary(MIN_WORD_FREQ)
vocab.build(captions)
print("✅ Vocabulary ready")

print("🔄 Loading model...")
model = DecoderRNN(
    EMBED_SIZE,
    HIDDEN_SIZE,
    len(vocab)
).to(DEVICE)

model.load_state_dict(
    torch.load(MODELS_DIR / "decoder.pth", map_location=DEVICE)
)
model.eval()
print("✅ Model loaded")

def generate_caption(max_len=20):
    word = torch.tensor([[vocab.stoi["<start>"]]]).to(DEVICE)
    caption_words = []

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(word)
            next_word = outputs[:, -1, :].argmax(dim=1).item()

            if next_word == vocab.stoi["<end>"]:
                break

            caption_words.append(vocab.itos[next_word])
            word = torch.cat(
                [word, torch.tensor([[next_word]]).to(DEVICE)], dim=1
            )

    return " ".join(caption_words)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    Image.open(io.BytesIO(image_bytes))  # image only for UI

    caption = generate_caption()
    return {"caption": caption}