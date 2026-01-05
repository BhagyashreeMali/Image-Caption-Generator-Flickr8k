import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import FlickrDataset, collate_fn
from .model import DecoderRNN
from .config import (
    IMAGES_DIR,
    CAPTIONS_FILE,
    VOCAB_PATH,
    MODELS_DIR,
    DEVICE,
    EMBED_SIZE,
    HIDDEN_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from .utils import load_pickle
from .preprocessing import Vocabulary


def train():
    # Load vocabulary
    vocab = load_pickle(VOCAB_PATH)

    # Dataset and DataLoader
    dataset = FlickrDataset(IMAGES_DIR, CAPTIONS_FILE, vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = DecoderRNN(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(vocab),
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Training started")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for images, captions, lengths in loader:
            captions = captions.to(DEVICE)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(inputs)

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "decoder.pth")
    print("✅ Model saved")


if __name__ == "__main__":
    train()