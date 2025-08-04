import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # (B, E, H', W')
        x = x.flatten(2)              # (B, E, N)
        x = x.transpose(1, 2)         # (B, N, E)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 embed_dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding
        num_tokens = 1 + self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)
        x = self.patch_embed(x)       # (B, N, E)

        # Prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, 1+N, E)

        # Add positional embedding and dropout
        x = self.dropout(x + self.pos_embed[:, :x.size(1), :])

        # Transformer encoding
        x = self.encoder(x)

        # Take cls token for classification
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)