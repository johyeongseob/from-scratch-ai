import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # load image
# img = Image.open("1200px-Cat03.jpg")
# img.show()
#
# transforms = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor()
# ])
#
# x = transforms(img)
# x = x[None, ...]
# print(f"x.shape: {x.shape}")  # [1, 3, 224, 224]
#
# patch_size = 16
# patches = rearrange(
#     x,
#     'b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
#     s1=patch_size,
#     s2=patch_size
# )
# # print(f"patches.shape: {patches.shape}")  # [1, 196, 768]
#
# patches_d = rearrange(
#     x,
#     'b c (h s1) (w s2) -> b h w s1 s2 c',
#     s1 = 16,
#     s2 = 16
# )
# print(f"patches_d.shape: {patches_d.shape}")  # [1, 14, 14, 16, 16, 3]

# fig, axes = plt.subplots(nrows=14, ncols=14, figsize=(20, 20))
# for i, j in itertools.product(range(14), repeat=2):
#     axes[i, j].imshow(patches_d[0, i, j])
#     axes[i, j].axis('off')
#     axes[i, j].set_title(f"patch ({i},{j})")
# fig.tight_layout()
# plt.show()

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()

        assert img_size / patch_size % 1 == 0, "img_size must be integer multiple of patch_size"

        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # [1, 1, 768]

        self.positional_emb = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))  # [197, 768]

    def forward(self, x):
        B, *_ = x.shape  # B: num of samples
        x = self.projection(x)  # [1, 196, 768]
        cls_token = repeat(self.cls_token, '() p e -> b p e', b = B)  # [1, 1, 768]

        x = torch.cat([cls_token, x], dim=1)
        x += self.positional_emb
        return x

# patch_embedding = PatchEmbedding()(x)
# print(f"patch_embedding.shape: {patch_embedding.shape}")  # [1, 197, 768]

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

        self.projection = nn.Linear(emb_size, emb_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x, mask=None):
        rearrange_heads = 'batch seq_len (num_head h_dim) -> batch num_head seq_len h_dim'
        # [batch, seq_len, **emb_size**] → [batch, **num_head**, seq_len, **h_dim**]

        queries = rearrange(self.query(x), rearrange_heads, num_head=self.num_heads)  # [Batch, heads, seq_len, h_dim]
        keys = rearrange(self.key(x), rearrange_heads, num_head=self.num_heads)  # [Batch, heads, seq_len, h_dim]
        values = rearrange(self.value(x), rearrange_heads, num_head=self.num_heads)  # [Batch, heads, seq_len, h_dim]

        # sum up over the last axis
        energies = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(energies.dtype).min  # -infinity (inf)
            energies.mask_fill(~mask, fill_value)

        attention = F.softmax(energies * self.scaling, dim=-1)
        attention = self.attn_dropout(attention)

        # sum up over the third axis
        out = torch.einsum('bhas, bhsd -> bhad', attention, values)
        out = rearrange(out, 'batch num_head seq_length dim -> batch seq_length (num_head dim)')
        out = self.projection(out)

        return out


# wrapper to perform the residual addition:
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):  # keyword arguments: 유연한 함수 인자 전달 방식
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

FeedForwardBlock=lambda emb_size=768, expansion=4, drop_p=0.: nn.Sequential(
    nn.Linear(emb_size, expansion * emb_size),
    nn.GELU(),
    nn.Dropout(drop_p),
    nn.Linear(expansion * emb_size, emb_size)
)


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0, **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
            ),

            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super(TransformerEncoder, self).__init__(
            *(TransformerEncoderBlock(**kwargs) for _ in range(depth))
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, num_classes=1000):
        super(ClassificationHead, self).__init__(
            Reduce('batch_size seq_len emb_dim -> batch_size emb_dim', reduction='mean'),  # mean pooling head
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )


class ViT(nn.Sequential):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, num_classes=1000, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size,),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )

# print(f"ViT()(x).shape: {ViT()(x).shape}")

# model = ViT()
# model = model.to(device)
# summary(model, input_size=(3, 224, 224), device=str(device))

