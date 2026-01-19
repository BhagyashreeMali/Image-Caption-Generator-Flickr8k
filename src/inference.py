import torch
from .model import DecoderRNN
from .config import *
from .utils import load_pickle
from .preprocessing import Vocabulary


def generate_caption(max_len=20):
    vocab = load_pickle(VOCAB_PATH)

    model = DecoderRNN(
        EMBED_SIZE,
        HIDDEN_SIZE,
        len(vocab),
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODELS_DIR / "decoder.pth", map_location=DEVICE)
    )
    model.eval()

    word = torch.tensor([[vocab.stoi["<start>"]]]).to(DEVICE)
    caption_words = []

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(word)
            next_word_id = outputs[:, -1, :].argmax(dim=1).item()

            if next_word_id == vocab.stoi["<end>"]:
                break

            caption_words.append(vocab.itos[next_word_id])
            word = torch.cat(
                [word, torch.tensor([[next_word_id]]).to(DEVICE)], dim=1
            )

    caption = " ".join(caption_words)
    return caption   # ✅ RETURN, not print


if __name__ == "__main__":
    caption = generate_caption()
    print("📝 Generated Caption:")
    print(caption) 