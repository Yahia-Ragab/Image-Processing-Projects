"""
vanilla_vit_debug.py

Vision Transformer (ViT-B/16) with detailed PyCharm breakpoints for debugging snapshots.
Each breakpoint matches the mandatory snapshot requirements (Input, Embedding, Attention, etc.).
Configuration:
- Patch size: 16
- Embedding dim: 768
- 12 heads
- 12 encoder blocks
- 1000 ImageNet classes
Compatible with PyTorch 2.8.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        breakpoint()  #1: Raw input image tensor (after preprocessing)
        patches = self.proj(x)
        breakpoint()  #2: Image divided into patches (before flattening)
        patches = patches.flatten(2).transpose(1, 2)
        breakpoint()  #3: Flattened patches (reshaped into vectors)
        return patches

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        breakpoint()  #16: Feed-forward input
        hidden = self.act(self.fc1(x))
        breakpoint()  #17: Feed-forward hidden layer output
        out = self.fc2(hidden)
        breakpoint()  #18: Feed-forward output after second linear
        out = self.drop(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        breakpoint()  #9: Multi-head attention queries (Q)
        breakpoint()  #10: Multi-head attention keys (K)
        breakpoint()  #11: Multi-head attention values (V)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        breakpoint()  #12: Attention scores before softmax
        attn = attn.softmax(dim=-1)
        breakpoint()  #13: Attention scores after softmax

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        breakpoint()  #14: Multi-head attention output (after concatenation)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        breakpoint()  #8: Encoder block input tensor
        x = x + self.attn(self.norm1(x))
        breakpoint()  #15: Residual connection + normalization (post-attention)
        x = x + self.mlp(self.norm2(x))
        breakpoint()  #19: Residual connection + normalization (post-MLP)
        breakpoint()  #20: Encoder block final output
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        breakpoint()  #5: Class token before concatenation
        x = torch.cat((cls_token, x), dim=1)
        breakpoint()  #6: Embeddings after adding the class token
        x = x + self.pos_embed
        breakpoint()  #7: Embeddings after adding positional encoding
        x = self.pos_drop(x)

        x = self.blocks[0](x)
        breakpoint()  #20: Encoder block 1 output

        for i in range(1, len(self.blocks) - 1):
            x = self.blocks[i](x)

        breakpoint()  #21: Encoder block 2 output
        x = self.blocks[-1](x)
        breakpoint()  #22: Encoder block N (last block) output

        x = self.norm(x)
        breakpoint()  #23: Final sequence output (including class token)
        cls = x[:, 0]
        breakpoint()  #24: Class token extracted (final representation)
        logits = self.head(cls)
        breakpoint()  #25: Classification head logits
        probs = F.softmax(logits, dim=-1)
        breakpoint()  #26: Softmax probabilities (example slice)
        return probs

def vit_base_patch16_224_debug():
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=1000)

if __name__ == "__main__":
    # --- Input Image Preprocessing ---
    img_path = "image0.jpg"  # Replace with your image file
    img = Image.open(img_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    x = preprocess(img).unsqueeze(0)

    model = vit_base_patch16_224_debug()
    output = model(x)
