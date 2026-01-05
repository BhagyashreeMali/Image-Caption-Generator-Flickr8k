from nltk.translate.bleu_score import sentence_bleu
from .inference import generate_caption
from .config import CAPTIONS_FILE
from .preprocessing import Vocabulary


def evaluate():
    references = []

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50:  # small sample
                break
            _, caption = line.strip().split("\t")
            references.append(caption.lower().split())

    hypothesis = generate_caption().lower().split()

    score = sentence_bleu(references, hypothesis)
    print("📊 BLEU Score:", score)


if __name__ == "__main__":
    evaluate()