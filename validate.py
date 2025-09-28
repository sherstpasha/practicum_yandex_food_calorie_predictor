import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
import timm
from torchmetrics import MeanAbsoluteError
from transformers import AutoTokenizer

from dataset import CaloriesDataset
from model import CaloriesModel


def validate(config, device, model_path):
    dish_df = pd.read_csv(config.DISH_CSV_PATH)
    ingredients_df = pd.read_csv(config.INGREDIENTS_CSV_PATH)

    id2ingr = dict(zip(ingredients_df["id"], ingredients_df["ingr"]))

    def ids_to_names(ingredients_str):
        ids = [int(x.replace("ingr_", "")) for x in ingredients_str.split(";")]
        names = [id2ingr[i] for i in ids if i in id2ingr]
        return " ".join(names)

    dish_df["ingredients_text"] = dish_df["ingredients"].apply(ids_to_names)
    df_test = dish_df[dish_df["split"] == "test"].copy()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    img_size = cfg.input_size[1:]
    test_transform = T.Compose(
        [
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std, inplace=True),
        ]
    )
    dataset_test = CaloriesDataset(
        df_test,
        config.IMAGES_PATH,
        tokenizer=tokenizer,
        transform=test_transform,
        shuffle_text=False,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    model = CaloriesModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mae_metric = MeanAbsoluteError().to(device)
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader_test, desc="Validation")):
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

            mae_metric.update(preds, total_calories)

            for i in range(len(preds)):
                row = df_test.iloc[batch_idx * config.BATCH_SIZE + i]
                results.append(
                    {
                        "pred": float(preds[i].cpu().item()),
                        "target": float(total_calories[i].cpu().item()),
                        "mass": float(total_mass[i].cpu().item()),
                        "ingredients": row["ingredients_text"],
                        "image_path": os.path.join(
                            config.IMAGES_PATH, row["dish_id"], "rgb.png"
                        ),
                    }
                )

    mae = mae_metric.compute().item()
    return results, mae
