class Config:
    SEED = 42
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 120

    MASS_EMB_DIM = 64
    IMAGE_MODEL_NAME = "efficientnet_lite0"
    TEXT_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
    IMAGE_EMB_DIM = 256
    TEXT_EMB_DIM = 256
    NUM_OUTPUTS = 1

    DISH_CSV_PATH = r"dataset\data\dish.csv"
    INGREDIENTS_CSV_PATH = r"dataset\data\ingredients.csv"
    IMAGES_PATH = r"dataset\data\images"
    SAVE_PATH = r"best_model.pth"
