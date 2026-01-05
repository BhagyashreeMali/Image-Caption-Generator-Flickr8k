from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class FlickrDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, max_len=30):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_len = max_len

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.image_ids = []
        self.captions = []

        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                img_cap, caption = line.strip().split("\t")

                # remove #0, #1 etc
                image_name = img_cap.split("#")[0].strip()

                self.image_ids.append(image_name)
                self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]

        # Fix suffix issue (handle cases like ".jpg.1" or extra extensions)
        if ".jpg" in image_name:
            image_name = image_name.split(".jpg")[0] + ".jpg"

        image_path = self.images_dir / image_name

        # Skip missing images safely (wrap around to next index)
        if not image_path.exists():
            return self.__getitem__((idx + 1) % len(self))

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tokens = [self.vocab.stoi["<start>"]]
        tokens += self.vocab.numericalize(self.captions[idx])
        tokens.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(tokens)


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    # Return images, padded captions, and lengths (useful for packing in RNNs)
    return images, padded, lengths