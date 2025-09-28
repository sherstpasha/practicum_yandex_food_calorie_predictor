import torch
import torch.nn as nn
import timm
from transformers import AutoModel


class CaloriesModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME, pretrained=True, features_only=True
        )
        out_channels = self.image_model.feature_info[-1]["num_chs"]
        self.image_proj = nn.Sequential(
            nn.Linear(out_channels, config.IMAGE_EMB_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.IMAGE_EMB_DIM),
        )

        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        text_hidden_dim = self.text_model.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, config.TEXT_EMB_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.TEXT_EMB_DIM),
        )

        fusion_dim = config.IMAGE_EMB_DIM + config.TEXT_EMB_DIM + 1

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, config.NUM_OUTPUTS),
        )

    def forward(self, total_mass, image, input_ids, attention_mask):
        features = self.image_model(image)[-1]
        features = features.mean(dim=[2, 3])
        img_emb = self.image_proj(features)

        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state.mean(dim=1)
        text_emb = self.text_proj(text_emb)

        mass_scalar = total_mass.unsqueeze(1)

        fused = torch.cat([img_emb, text_emb, mass_scalar], dim=1)

        return self.regressor(fused)
