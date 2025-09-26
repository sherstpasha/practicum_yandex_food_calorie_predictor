class Config:
    SEED = 42
    IMAGE_MODEL_NAME = "efficientnet_b2"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head"

    BATCH_SIZE = 64
    LR_IMAGE = 3e-5
    LR_FEATURES = 1e-3
    LR_HEAD = 3e-4

    EPOCHS = 100

    N_FEATURES_HIDDEN_DIM = 556
    FEATURES_HIDDEN_DIM = 2048
    HEAD_HIDDEN_DIM = 1024
    NUM_OUTPUTS = 1

    DISH_CSV_PATH = r"dataset\data\dish.csv"
    INGREDIENTS_CSV_PATH = r"dataset\data\ingredients_ru.csv"
    IMAGES_PATH = r"dataset\data\images"
    SAVE_PATH = r"best_model.pth"

cfg = Config()
