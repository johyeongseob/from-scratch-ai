import torch
from torch.utils.data import Dataset
import config
from PIL import Image


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames  # Each image is repeated 5 times for 5 captions in captions.csv
        self.captions = list(captions)  # class: 'pandas.core.series.Series' -> 'list'
        self.encoded_captions = tokenizer(  # Encode all captions from the CSV file using the tokenizer
            list(captions),
            padding=True,
            truncation=True,
            max_length=config.max_length
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {}

        # Extract the encoded token (e.g., input_ids, attention_mask) for the current index
        for key, values in self.encoded_captions.items():
            item[key] = torch.tensor(values[idx])

        image = Image.open(f"{config.image_path}/{self.image_filenames[idx]}").convert("RGB")  # directory + filename
        item['image'] = self.transforms(image)
        item['caption'] = self.captions[idx]

        return item
