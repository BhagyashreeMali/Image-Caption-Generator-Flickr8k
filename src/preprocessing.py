import nltk
from collections import Counter
from pathlib import Path
import argparse

from .utils import save_pickle
from .config import CAPTIONS_FILE, VOCAB_PATH, MIN_WORD_FREQ

nltk.download("punkt")
nltk.download("punkt_tab")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return nltk.word_tokenize(text.lower())

    def build(self, sentences):
        frequencies = Counter()
        idx = 4

        for sentence in sentences:
            frequencies.update(self.tokenize(sentence))

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in self.tokenize(text)]


def load_captions(path):
    captions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            _, cap = line.strip().split("\t")
            captions.append(cap)
    return captions


def build_and_save_vocab():
    captions = load_captions(CAPTIONS_FILE)
    vocab = Vocabulary(MIN_WORD_FREQ)
    vocab.build(captions)

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(vocab, VOCAB_PATH)

    print(f"✅ Vocab saved | size={len(vocab)}")


if __name__ == "__main__":
    build_and_save_vocab()