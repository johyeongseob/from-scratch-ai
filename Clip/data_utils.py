import torch
import numpy as np
import pandas as pd
import config
from CLIPDataset import CLIPDataset
from torch.utils.data import Dataset
from torchvision import transforms


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{config.captions_path}")
    max_id = dataframe["id"].max() + 1 if not config.debug else 100
    image_ids = np.arange(0, max_id)  # 31782
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


# You can add augmentations in the train mode if needed
def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((config.size, config.size)),
                transforms.ToTensor()
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((config.size, config.size)),
                transforms.ToTensor()
            ]
        )
