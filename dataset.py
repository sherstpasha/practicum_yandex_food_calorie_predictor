from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
from torchvision.transforms import Resize
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomRotation
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import timm
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch

cfg = timm.get_pretrained_cfg("efficientnet_b2")




class CaloriesDataset(Dataset):
    def __init__(self, dish_df, dish_images_path, ingredients_df, tranforms=None):
        self.dish_df = dish_df
        self.dish_images_path = dish_images_path

        self.mass_min = self.dish_df["total_mass"].min()
        self.mass_max = self.dish_df["total_mass"].max()

        all_ids = ingredients_df["id"].tolist()
        self.mlb = MultiLabelBinarizer(classes=all_ids)
        self.mlb.fit([all_ids])

        self.tranforms = tranforms

    def __len__(self):
        return len(self.dish_df)

    def __getitem__(self, idx):
        dish_id = self.dish_df.iloc[idx]["dish_id"]
        image_path = os.path.join(self.dish_images_path, dish_id, "rgb.png")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.tranforms:
            img_rgb = self.tranforms(img_rgb)

        total_calories = self.dish_df.iloc[idx]["total_calories"]
        total_mass = self.dish_df.iloc[idx]["total_mass"]
        total_mass = (total_mass - self.mass_min) / (self.mass_max - self.mass_min)
        ingredients = self.dish_df.iloc[idx]["ingredients"]

        ingredients_ids = [
            int(token.replace("ingr_", "").lstrip("0"))
            for token in ingredients.split(";")
        ]
        ingredients_vector = self.mlb.transform([ingredients_ids])[0]
        features = np.concatenate([ingredients_vector, [total_mass]]).astype(np.float32)

        return img_rgb, torch.tensor(features, dtype=torch.float32), torch.tensor(total_calories, dtype=torch.float32)


def decode_features(features, ingredients_df):
    ingr_vector = features[:-1]
    mass = features[-1]

    ingr_ids = [i + 1 for i, v in enumerate(ingr_vector) if v == 1]

    id2name = dict(zip(ingredients_df["id"], ingredients_df["ingr"]))
    ingr_names = [id2name[i] for i in ingr_ids]

    return ", ".join(ingr_names), mass
