import torch
import torch.nn as nn
import numpy as np

class PreNorm(nn.Module):
    """
    PreNorm
    Goal: Layer normalization before the function
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    fn: function --- Function
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormCross(nn.Module):
    """
    PreNormCross
    Goal: Layer normalization before the function
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    fn: function --- Function
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2),**kwargs)

class PatchEmbedding(nn.Module):
    """
    PatchEmbedding
    Goal: Patch Embedding
    ----------------
    Parameters:
    -----------
    image_size: int --- Size of the image
    patch_size: int --- Size of the patch
    in_channel: int --- Number of input channels
    embed_dim: int --- Embedding dimension
    """
    def __init__(self, image_size=256, patch_size=16, in_channel=3, embed_dim=768):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        
        self.num_patches = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class UnPatchEmbedding(nn.Module):
    """
    UnPatchEmbedding
    Goal: UnPatch Embedding
    ----------------
    Parameters:
    -----------
    patch_size: int --- Size of the patch
    in_channel: int --- Number of input channels
    embed_dim: int --- Embedding dimension
    """
    def __init__(self, patch_size=16, in_channel=3, embed_dim=768):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        # self.unflatten = nn.Unflatten(2, (patch_size, patch_size))
        self.proj = nn.ConvTranspose2d(768, 3, kernel_size=16, stride=16)
    
    def forward(self, x: torch.Tensor):
        x = x[:, 1:, :]
        x = x.transpose(1, 2)
        # print(x.shape)
        # x = self.unflatten(x)
        x = x.unflatten(2, (16, 16))
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """
    FeedForward
    Goal: Feed Forward Network
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    hidden_dim: int --- Hidden layer dimension
    dropout: float --- Dropout value
    recons: bool --- Reconstruction
    """
    def __init__(self, dim, hidden_dim=None, dropout = 0., recons=False):
        super().__init__()

        if hidden_dim == None:
            hidden_dim = dim
        
        if recons == False:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.GELU()
            )
    def forward(self, x):
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    """
    MultiHeadSelfAttention
    Goal: Multi Head Self Attention
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    n_heads: int --- Number of heads
    qkv_bias: bool --- Query, Key, Value bias
    attn_dropout: float --- Attention dropout
    proj_dropout: float --- Projection dropout
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
    
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 

        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Key Transpose
        k_t = k.transpose(-2, -1)
        
        # Note: Softmax of (Query * Key Transpose * squareroot of head dimension)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  

        weighted_avg = weighted_avg.transpose(1, 2) 
        weighted_avg = weighted_avg.flatten(2)  

        x = self.proj(weighted_avg) 
        x = self.proj_drop(x)

        return x

class MultiHeadCrossAttention(nn.Module):
    """
    MultiHeadCrossAttention
    Goal: Multi Head Cross Attention
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    n_heads: int --- Number of heads
    qkv_bias: bool --- Query, Key, Value bias
    attn_dropout: float --- Attention dropout
    proj_dropout: float --- Projection dropout
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
    
    def forward(self, x1, x2):
        
        n_samples1, n_tokens1, dim1 = x1.shape

        n_samples2, n_tokens2, dim2 = x2.shape

        if dim1 != dim2:
            raise ValueError
        
        if dim1 != self.dim:
            raise ValueError

        qkv1 = self.qkv1(x1)
        qkv2 = self.qkv2(x2)

        qkv1 = qkv1.reshape(n_samples1, n_tokens1, 1, self.n_heads, self.head_dim)
        qkv1 = qkv1.permute(2, 0, 3, 1, 4) 
        
        qkv2 = qkv2.reshape(n_samples2, n_tokens2, 2, self.n_heads, self.head_dim)
        qkv2 = qkv2.permute(2, 0, 3, 1, 4) 

        q, k, v = qkv1[0], qkv2[0], qkv2[1]
        
        # Key Transpose
        k_t = k.transpose(-2, -1)
        
        # Note: Softmax of (Query * Key Transpose * squareroot of head dimension)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  

        weighted_avg = weighted_avg.transpose(1, 2) 
        weighted_avg = weighted_avg.flatten(2)  

        x = self.proj(weighted_avg) 
        x = self.proj_drop(x)

        return x

class Transformer(nn.Module):
    """
    Transformer
    Goal: Transformer
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    depth: int --- Depth of the transformer
    mlp_dim: int --- Hidden layer dimension
    n_heads: int --- Number of heads
    qkv_bias: bool --- Query, Key, Value bias
    attn_dropout: float --- Attention dropout
    proj_dropout: float --- Projection dropout
    """
    def __init__(self, dim, depth, mlp_dim=None, n_heads=12, qkv_bias=True, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, MultiHeadSelfAttention(dim, n_heads, qkv_bias, attn_dropout, proj_dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=proj_dropout))
                ])
            )
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossTransformer(nn.Module):
    """
    CrossTransformer
    Goal: Cross Transformer
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    depth: int --- Depth of the transformer
    mlp_dim: int --- Hidden layer dimension
    n_heads: int --- Number of heads
    qkv_bias: bool --- Query, Key, Value bias
    attn_dropout: float --- Attention dropout
    proj_dropout: float --- Projection dropout
    mode: str --- Mode of the cross transformer
    """
    def __init__(self, dim, depth, mlp_dim=None, n_heads=12, qkv_bias=True, attn_dropout=0., proj_dropout=0., mode="x1"):
        super().__init__()
        self.depth = depth
        self.mode = mode
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNormCross(dim, MultiHeadCrossAttention(dim, n_heads, qkv_bias, attn_dropout, proj_dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=proj_dropout))
                ])
            )
    
    def forward(self, x1, x2):
        d = 0 
        x = x1
        for attn, ff in self.layers:
            if self.mode == 'x1':
                x1 = attn(x1, x2) + x1
                x1 = ff(x1) + x1
                x = x1
            elif self.mode == 'x2':
                x2 = attn(x1, x2) + x2
                x2 = ff(x1) + x2
                x = x2
            else:
                if d % 2 == 0:
                    x1 = attn(x1, x2) + x1
                    x1 = ff(x1) + x1
                    x = x1
                else:
                    x2 = attn(x1, x2) + x2
                    x2 = ff(x1) + x2
                    x = x2
            d = d + 1
        return x

class MultiScaleTransformer(nn.Module):
    """
    MultiScaleTransformer
    Goal: Multi Scale Transformer
    ----------------
    Parameters:
    -----------
    dim: int  --- Embedding dimension
    depth_1: int --- Depth of the transformer for image 1
    depth_2: int --- Depth of the transformer for image 2
    cross_depth: int --- Depth of the cross transformer
    mlp_dim_1: int --- Hidden layer dimension for image 1
    n_heads_1: int --- Number of heads for image 1
    qkv_bias_1: bool --- Query, Key, Value bias for image 1
    attn_dropout_1: float --- Attention dropout for image 1
    proj_dropout_1: float --- Projection dropout for image 1
    mlp_dim_2: int --- Hidden layer dimension for image 2
    n_heads_2: int --- Number of heads for image 2
    qkv_bias_2: bool --- Query, Key, Value bias for image 2
    attn_dropout_2: float --- Attention dropout for image 2
    proj_dropout_2: float --- Projection dropout for image 2
    mlp_dim_cross: int --- Hidden layer dimension for cross transformer
    n_heads_cross: int --- Number of heads for cross transformer
    qkv_bias_cross: bool --- Query, Key, Value bias for cross transformer
    attn_dropout_cross: float --- Attention dropout for cross transformer
    proj_dropout_cross: float --- Projection dropout for cross transformer
    mode: str --- Mode of the cross transformer
    """
    def __init__(self, dim, depth_1, depth_2, cross_depth, mlp_dim_1=None, n_heads_1=12, qkv_bias_1=True, attn_dropout_1=0., proj_dropout_1=0.,
                 mlp_dim_2=None, n_heads_2=12, qkv_bias_2=True, attn_dropout_2=0., proj_dropout_2=0., 
                 mlp_dim_cross=None, n_heads_cross=12, qkv_bias_cross=True, attn_dropout_cross=0., proj_dropout_cross=0.,mode="x1"):
        super().__init__()
        self.transformer_x1 = Transformer(dim, depth_1, mlp_dim_1, n_heads_1, qkv_bias_1, attn_dropout_1, proj_dropout_1)
        self.transformer_x2 = Transformer(dim, depth_2, mlp_dim_2, n_heads_2, qkv_bias_2, attn_dropout_2, proj_dropout_2)
        self.crosstransformer = CrossTransformer(dim, cross_depth, mlp_dim_cross, n_heads_cross, qkv_bias_cross, attn_dropout_cross, proj_dropout_cross, mode)

    def forward(self, x1, x2):
        x1 = self.transformer_x1(x1)
        x2 = self.transformer_x1(x2)
        x3 = self.crosstransformer(x1, x2)
        x4 = self.crosstransformer(x2, x1)
        x = x3 + x4
        return x

class ReconstructBlock(nn.Module):
    """
    ReconstructBlock
    Goal: Reconstruct Block
    ----------------
    Parameters:
    -----------
    dim: int --- Embedding dimension
    hidden_dim: int --- Hidden layer dimension
    dropout: float --- Dropout value
    recons: bool --- Reconstruction
    patch_size: int --- Size of the patch
    in_channel: int --- Number of input channels
    embed_dim: int --- Embedding dimension
    """
    def __init__(self, dim, hidden_dim=None, dropout = 0., recons=True, patch_size=16, in_channel=3, embed_dim=768):
        super().__init__()
        self.mlp = FeedForward(dim, hidden_dim, dropout, recons)
        self.unpatch = UnPatchEmbedding(in_channel, embed_dim, patch_size)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.unpatch(x)
        return x

class CrossGenViT(nn.Module):
    """
    CrossGenViT
    Goal: Cross Generative Vision Transformer
    ----------------
    Parameters:
    -----------
    depth_1: int
        Depth of the transformer for image 1
    depth_2: int
        Depth of the transformer for image 2
    cross_depth: int
        Depth of the cross transformer
    multi_scale_depth: int
        Depth of the multi scale transformer
    image_size: int
        Size of the image
    patch_size: int
        Size of the patch
    in_channel: int
        Number of input channels
    embed_dim: int
        Embedding dimension
    emb_dropout: float
        Dropout value
    mlp_dim_1: int
        Hidden layer dimension for image 1
    n_heads_1: int
        Number of heads for image 1
    qkv_bias_1: bool
        Query, Key, Value bias for image 1
    attn_dropout_1: float
        Attention dropout for image 1
    proj_dropout_1: float
        Projection dropout for image 1
    mlp_dim_2: int
        Hidden layer dimension for image 2
    n_heads_2: int
        Number of heads for image 2
    qkv_bias_2: bool
        Query, Key, Value bias for image 2
    attn_dropout_2: float
        Attention dropout for image 2
    proj_dropout_2: float
        Projection dropout for image 2
    mlp_dim_cross: int
        Hidden layer dimension for cross transformer
    n_heads_cross: int
        Number of heads for cross transformer
    qkv_bias_cross: bool
        Query, Key, Value bias for cross transformer
    attn_dropout_cross: float
        Attention dropout for cross transformer
    proj_dropout_cross: float
        Projection dropout for cross transformer
    mode: str
        Mode of the cross transformer
    """
    def __init__(self, depth_1=3, depth_2=3, cross_depth=3, multi_scale_depth=3, image_size=256, patch_size=16, in_channel=3, embed_dim=768, emb_dropout=0., mlp_dim_1=None, n_heads_1=12, qkv_bias_1=True, attn_dropout_1=0., proj_dropout_1=0.,
                 mlp_dim_2=None, n_heads_2=12, qkv_bias_2=True, attn_dropout_2=0., proj_dropout_2=0., 
                 mlp_dim_cross=None, n_heads_cross=12, qkv_bias_cross=True, attn_dropout_cross=0., proj_dropout_cross=0.,mode="x1"):
        super().__init__()

        self.patch = PatchEmbedding(image_size, patch_size, in_channel, embed_dim)
        
        self.cls_token_x1 = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding_x1 = nn.Parameter(torch.randn(1, self.patch.num_patches + 1, embed_dim))
        self.dropout_x1 = nn.Dropout(emb_dropout)
        
        self.cls_token_x2 = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding_x2 = nn.Parameter(torch.randn(1, self.patch.num_patches + 1, embed_dim))
        self.dropout_x2 = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_depth):
            self.multi_scale_transformers.append(
                MultiScaleTransformer(embed_dim, depth_1, depth_2, cross_depth, mlp_dim_1, n_heads_1, qkv_bias_1, attn_dropout_1, proj_dropout_1,
                                    mlp_dim_2, n_heads_2, qkv_bias_2, attn_dropout_2, proj_dropout_2, 
                                    mlp_dim_cross, n_heads_cross, qkv_bias_cross, attn_dropout_cross, proj_dropout_cross,mode
                                    )
            )
        self.reconstruct = ReconstructBlock(embed_dim, embed_dim, emb_dropout, patch_size, in_channel, embed_dim)
    
    def forward(self, img1, img2):
        batch_1 = img1.shape[0]
        batch_2 = img2.shape[0]

        x1 = self.patch(img1)
        x2 = self.patch(img2)

        cls_token_x1 = self.cls_token_x1.expand(batch_1, -1, -1)
        x1 = torch.cat((cls_token_x1, x1), dim=1)
        x1 = x1 + self.pos_embedding_x1
        x1 = self.dropout_x1(x1)
        
        cls_token_x2 = self.cls_token_x2.expand(batch_2, -1, -1)
        x2 = torch.cat((cls_token_x2, x2), dim=1)
        x2 = x2 + self.pos_embedding_x2
        x2 = self.dropout_x2(x2)
        x = x1
        for multi_scale_transformer in self.multi_scale_transformers:
            x = multi_scale_transformer(x, x2)
        x = self.reconstruct(x)
        return x

if __name__ == "__main__":
    
    # Example with prefilled parameters for testing the model
    # Small sample code which is used in Unpaired Image Dehazing GAN

    # Test the model
    img1 = torch.ones([1, 3, 256, 256])
    img2 = torch.ones([1, 3, 256, 256])


    model = CrossGenViT(embed_dim=768, depth_1=3, depth_2=3, cross_depth=3)
    output = model(img1, img2)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    print("Shape of out :", output.shape)

"""
Output:
-------
Trainable Parameters: 99.107M
Shape of out : torch.Size([1, 3, 256, 256])
"""

"""
Possible Improvements:
----------------------
1. CrossViT Generator is great for global context of the image but it cannot capture local features. 
2. We can use a combination of CrossViT and Convolutional Neural Network to capture both global and local features.
3. We can try different type of attention mechanism (only the example: Axial Attention) to capture the global context as well as local features of the image.
4. For research purposes, I have used CrossViT as a generator as well as a discriminator in the GAN model. Finally, I have used the Attention U-Net based as a generator to capture local features.
5. CloVe model (in during the interview) can be used to capture the local features of the image for image generation tasks.
"""