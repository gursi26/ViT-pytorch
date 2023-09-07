from torch import nn
import torch

class ViTLinearProjection(nn.Module):

    def __init__(self, in_channels, patch_size, d_model):
        super(ViTLinearProjection, self).__init__()
        self.linear = nn.Linear(in_channels * patch_size * patch_size, d_model)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return self.linear(x)

class FeedForward(nn.Module):

    def __init__(self, input_dim):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, x):
        return self.layer(x)


class ViTEncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, p=0.1):
        super(ViTEncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=p)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        skip_x = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x)
        x = x + skip_x
        skip_x = x
        x = self.ln2(x)
        x = self.mlp(x)
        return x + skip_x
    

class ViT(nn.Module):

    def __init__(self, in_channels, patch_size, img_size, num_classes, d_model, num_heads, n_layers, p=0.1):
        super(ViT, self).__init__()
        self.in_proj = ViTLinearProjection(in_channels, patch_size, d_model)
        self.layers = nn.ModuleList([
            ViTEncoderBlock(d_model, num_heads, p) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, d_model))

    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.position_embedding
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x[:, 0, :])