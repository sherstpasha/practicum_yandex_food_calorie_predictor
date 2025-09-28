from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import random


class CaloriesDataset(Dataset):
    def __init__(
        self,
        dish_df,
        images_path,
        tokenizer,
        transform=None,
        max_len=64,
        shuffle_text=False,
    ):
        self.dish_df = dish_df.reset_index(drop=True)
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.shuffle_text = shuffle_text

    def __len__(self):
        return len(self.dish_df)

    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]

        total_calories = torch.tensor(row["total_calories"], dtype=torch.float32)
        total_mass = torch.tensor(row["total_mass"], dtype=torch.float32)

        dish_id = row["dish_id"]
        img_path = os.path.join(self.images_path, dish_id, "rgb.png")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        ingredients_text = row["ingredients_text"]

        if self.shuffle_text:
            parts = [x.strip() for x in ingredients_text.split(",")]
            random.shuffle(parts)
            ingredients_text = ", ".join(parts)

        encoding = self.tokenizer(
            ingredients_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "total_calories": total_calories,
            "total_mass": total_mass,
            "image": img,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ingredients_text": ingredients_text,
        }


def collate_fn(batch):
    total_calories = torch.stack([item["total_calories"] for item in batch])
    total_mass = torch.stack([item["total_mass"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "total_calories": total_calories,
        "total_mass": total_mass,
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
