import timm
import torch

class CrossAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads=4):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(emb_dim)
    
    def forward(self, query, context):
        attn_out, _ = self.cross_attn(
            query=query,
            key=context,
            value=context
        )
        return self.norm(attn_out)

class CaloriesModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_model = timm.create_model(
                config.IMAGE_MODEL_NAME,
                pretrained=True,
                features_only=True
            )
        img_out_ch = self.image_model.feature_info[-1]["num_chs"]
        
        self.image_proj = torch.nn.Linear(img_out_ch, config.HEAD_HIDDEN_DIM)
        self.features_model = torch.nn.Sequential(
            torch.nn.Linear(config.N_FEATURES_HIDDEN_DIM, config.HEAD_HIDDEN_DIM),
            torch.nn.LayerNorm(config.HEAD_HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(config.HEAD_HIDDEN_DIM, config.HEAD_HIDDEN_DIM),
            torch.nn.LayerNorm(config.HEAD_HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )

        self.fusion = CrossAttention(emb_dim=config.HEAD_HIDDEN_DIM, num_heads=4)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(config.HEAD_HIDDEN_DIM, config.HEAD_HIDDEN_DIM // 2),
            torch.nn.LayerNorm(config.HEAD_HIDDEN_DIM // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(config.HEAD_HIDDEN_DIM // 2, config.NUM_OUTPUTS),
        )

    def forward(self, img_rgb, features):
        feat_map = self.image_model(img_rgb)[-1] # [B, C, H, W]

        img_tokens = feat_map.flatten(2).transpose(1, 2) # [B, H*W, C]
        img_tokens = self.image_proj(img_tokens)

        features_emb = self.features_model(features).unsqueeze(1)
        fused  = self.fusion(features_emb, img_tokens).squeeze(1)

        return self.regressor(fused)
