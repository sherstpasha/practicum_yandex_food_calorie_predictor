import os
import random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import CaloriesDataset
from model import CaloriesModel
from config import Config

import timm
from torchvision import transforms as T
from transformers import AutoTokenizer


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def train(config, device):
    print(f"Using device: {device}")
    writer = SummaryWriter(log_dir="runs/calories_mass_image_text")

    dish_df = pd.read_csv(config.DISH_CSV_PATH)
    ingredients_df = pd.read_csv(config.INGREDIENTS_CSV_PATH)

    id2ingr = dict(zip(ingredients_df["id"], ingredients_df["ingr"]))

    def ids_to_names(ingredients_str):
        ids = [int(x.replace("ingr_", "")) for x in ingredients_str.split(";")]
        names = [id2ingr[i] for i in ids if i in id2ingr]
        return " ".join(names)

    dish_df["ingredients_text"] = dish_df["ingredients"].apply(ids_to_names)

    df_train = dish_df[dish_df["split"] == "train"].copy()
    df_test = dish_df[dish_df["split"] == "test"].copy()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    img_size = cfg.input_size[1:]

    train_transform = T.Compose(
        [
            T.Resize(img_size, antialias=True),
            T.RandomRotation(degrees=5),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std, inplace=True),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std, inplace=True),
        ]
    )

    dataset_train = CaloriesDataset(
        df_train,
        config.IMAGES_PATH,
        tokenizer=tokenizer,
        transform=train_transform,
        shuffle_text=True,
    )
    dataset_test = CaloriesDataset(
        df_test,
        config.IMAGES_PATH,
        tokenizer=tokenizer,
        transform=test_transform,
        shuffle_text=False,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    seed_everything(config.SEED)
    model = CaloriesModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )
    criterion = torch.nn.MSELoss()

    mae_metric = MeanAbsoluteError().to(device)
    best_mae = float("inf")

    print("Training started...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss_train = 0.0
        mae_metric.reset()
        for batch in tqdm(
            dataloader_train, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False
        ):
            total_calories = batch["total_calories"].to(device)
            total_mass = batch["total_mass"].to(device)
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            preds = model(
                total_mass=total_mass,
                image=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).squeeze()
            loss = criterion(preds, total_calories)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            mae_metric.update(preds, total_calories)

        mae_train = mae_metric.compute().item()

        model.eval()
        total_loss_test = 0.0
        mae_metric.reset()
        with torch.no_grad():
            for batch in dataloader_test:
                total_calories = batch["total_calories"].to(device)
                total_mass = batch["total_mass"].to(device)
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                preds = model(
                    total_mass=total_mass,
                    image=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).squeeze()
                loss = criterion(preds, total_calories)
                total_loss_test += loss.item()
                mae_metric.update(preds, total_calories)

        mae_test = mae_metric.compute().item()
        scheduler.step()

        if mae_test < best_mae:
            best_mae = mae_test
            torch.save(model.state_dict(), config.SAVE_PATH)

        writer.add_scalar("Loss/train", total_loss_train / len(dataloader_train), epoch)
        writer.add_scalar("Loss/test", total_loss_test / len(dataloader_test), epoch)
        writer.add_scalar("MAE/train", mae_train, epoch)
        writer.add_scalar("MAE/test", mae_test, epoch)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} "
            f"| loss_train: {total_loss_train/len(dataloader_train):.4f} "
            f"| loss_test: {total_loss_test/len(dataloader_test):.4f} "
            f"| MAE Train {mae_train:.4f} "
            f"| MAE Valid {mae_test:.4f}"
        )

    writer.close()
    print(f"Best MAE: {best_mae:.4f}")
