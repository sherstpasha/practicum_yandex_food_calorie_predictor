import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms as T
import timm

from config import Config
from model import CaloriesModel

# --- инициализация ---
cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# загружаем модель
model = CaloriesModel(cfg).to(device)
model.load_state_dict(torch.load(cfg.SAVE_PATH, map_location=device))
model.eval()

# токенизатор
tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)

# трансформации для картинок (как в test_transform)
cfg_img = timm.get_pretrained_cfg(cfg.IMAGE_MODEL_NAME)
img_size = cfg_img.input_size[1:]
test_transform = T.Compose(
    [
        T.Resize(img_size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=cfg_img.mean, std=cfg_img.std, inplace=True),
    ]
)


# --- функция предсказания ---
def predict_calories(image, description, mass):
    if image is None or description.strip() == "":
        return "Загрузите фото и введите описание"

    # обрабатываем картинку
    img = Image.fromarray(image).convert("RGB")
    img = test_transform(img).unsqueeze(0).to(device)

    # токенизация текста
    encoding = tokenizer(
        description,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # масса
    total_mass = torch.tensor([float(mass)], dtype=torch.float32).to(device)

    # предсказание
    with torch.no_grad():
        pred = (
            model(
                total_mass=total_mass,
                image=img,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            .squeeze()
            .cpu()
            .item()
        )

    return f"Предсказанная калорийность: {pred:.1f} кКал"


# --- интерфейс ---
iface = gr.Interface(
    fn=predict_calories,
    inputs=[
        gr.Image(type="numpy", label="Фото блюда"),
        gr.Textbox(label="Описание / ингредиенты"),
        gr.Number(label="Масса (г)", value=100),
    ],
    outputs=gr.Textbox(label="Калорийность"),
    title="Нейросеть для предсказания калорийности блюд",
    description="Загрузите фото, введите ингредиенты и массу блюда — модель предскажет калорийность.",
)

if __name__ == "__main__":
    iface.launch()
