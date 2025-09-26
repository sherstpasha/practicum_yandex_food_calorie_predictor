import os
import random
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms import RandomRotation
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import GaussianBlur
from torchvision.transforms import ColorJitter
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import timm
import torch
import numpy as np
from torchmetrics import MeanAbsoluteError
from torch.utils.tensorboard import SummaryWriter
from dataset import CaloriesDataset, decode_features
from model import CaloriesModel
from config import Config



def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def set_requires_grad(module: torch.nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False

def train(config, device):
    print(device)
    writer = SummaryWriter(log_dir="runs/calories_experiment")

    cfg_image_model = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    train_transforms = Compose([
        ToTensor(),
        RandomResizedCrop(cfg_image_model.input_size[1:], scale=(0.9, 1.0), antialias=True),
        RandomRotation(degrees=10),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        Normalize(mean=cfg_image_model.mean, std=cfg_image_model.std, inplace=True),
    ])


    test_transforms = Compose(
        [
            ToTensor(),
            Resize(cfg_image_model.input_size[1:], antialias=True),
            Normalize(mean=cfg_image_model.mean, std=cfg_image_model.std, inplace=True),
        ]
    )

    seed_everything(config.SEED)
    model = CaloriesModel(config=config).to(device)

    set_requires_grad(model.image_model,
                        unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    optimizer = torch.optim.AdamW([{
            'params': model.image_model.parameters(),
            'lr': config.LR_IMAGE
        }, {
            'params': model.features_model.parameters(),
            'lr': config.LR_FEATURES
        }, {
            'params': model.regressor.parameters(),
            'lr': config.LR_HEAD
        }], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    criterion = torch.nn.MSELoss()

    DISH_CSV_PATH = config.DISH_CSV_PATH
    INGREDIENTS_CSV_PATH  = config.INGREDIENTS_CSV_PATH
    IMAGES_PATH = config.IMAGES_PATH

    dish_df = pd.read_csv(DISH_CSV_PATH)
    dish_df_train = dish_df[dish_df["split"]=='train']
    dish_df_test = dish_df[dish_df["split"]=='test']
    ingredients_df = pd.read_csv(INGREDIENTS_CSV_PATH)

    dataset_train = CaloriesDataset(
        dish_df=dish_df_train,
        dish_images_path=IMAGES_PATH,
        ingredients_df=ingredients_df,
        tranforms=train_transforms
    )

    dataset_test = CaloriesDataset(
        dish_df=dish_df_test,
        dish_images_path=IMAGES_PATH,
        ingredients_df=ingredients_df,
        tranforms=test_transforms
    )

    dataloader_train = DataLoader(dataset=dataset_train,
                                batch_size=config.BATCH_SIZE,
                                shuffle = True,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)
    
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle = False,
                                 num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)

    mae_metric = MeanAbsoluteError().to(device)
    best_mae = float("inf")

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss_train = 0.0
        mae_metric.reset()
        for batch in dataloader_train:
            img_rgb, features, total_calories = batch
            img_rgb, features, total_calories = img_rgb.to(device), features.to(device), total_calories.to(device)
            
            optimizer.zero_grad()
            logits = model(img_rgb=img_rgb, features=features).squeeze()
            loss = criterion(logits, total_calories)

            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            mae_metric.update(logits, total_calories)

        mae_train = mae_metric.compute().item()
        all_preds, all_targets, all_imgs, all_features = [], [], [], []
        total_loss_test = 0.0
        mae_metric.reset()
        model.eval()
        with torch.no_grad():
            for batch in dataloader_test:
                img_rgb, features, total_calories = batch
                img_rgb, features, total_calories = img_rgb.to(device), features.to(device), total_calories.to(device)

                logits = model(img_rgb=img_rgb, features=features).squeeze()
                loss = criterion(logits, total_calories)
                total_loss_test += loss.item()
                mae_metric.update(logits, total_calories)

                all_preds.extend(logits.cpu().numpy())
                all_targets.extend(total_calories.cpu().numpy())
                all_imgs.extend(img_rgb.cpu())
                all_features.extend(features.cpu().numpy())

        mae_test = mae_metric.compute().item()
        scheduler.step(mae_test)

        if mae_test < best_mae:
            best_mae = mae_test
            torch.save(model.state_dict(), cfg.SAVE_PATH)

        writer.add_scalar("Loss/train", total_loss_train/len(dataloader_train), epoch)
        writer.add_scalar("Loss/test", total_loss_test/len(dataloader_test), epoch)
        writer.add_scalar("MAE/train", mae_train, epoch)
        writer.add_scalar("MAE/test", mae_test, epoch)

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LR/group_{i}", param_group['lr'], epoch)

        if epoch % 10 == 0:
            idxs = random.sample(range(len(all_preds)), 10)
            for i, idx in enumerate(idxs):
                img = all_imgs[idx]
                pred = all_preds[idx]
                target = all_targets[idx]
                ingr, mass = decode_features(all_features[idx], ingredients_df)
                mass = mass * (dataset_train.mass_max - dataset_train.mass_min) + dataset_train.mass_min

                caption = f"Ingr: {ingr}\nMass: {mass:.1f} g\nTrue: {target:.1f} kcal | Pred: {pred:.1f} kcal"
                print(f"{caption}")
                print()

                writer.add_image(f"ValExample/{i}", img, epoch)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | loss_train: {total_loss_train/len(dataloader_train):.4f} | loss_test: {total_loss_test/len(dataloader_train):.4f} | MAE Train {mae_train:.4f} | MAE valid {mae_test:.4f}")

    writer.close()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    train(cfg, device)