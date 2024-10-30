import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

from typing import List, Optional, Callable, Tuple
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from functools import partial

from .classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# sinusoidal positions

def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, cond_fn = None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob = 0.,
        return_embeddings = False
    ):
        x = self.conv_stem(x)

        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers)

        for stage, cond_fn in zip(self.layers, cond_fns):
            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)

# attention

class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

@beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.1,
        ff_dropout = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers) * 2

        cond_fns = iter(cond_fns)

        for attn, ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns)) + x
             x = ff(x, cond_fn = next(cond_fns)) + x
        return x

@beartype
class FusionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.1,
        ff_dropout = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    def forward(
        self,
        x,
        context,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers) * 2

        cond_fns = iter(cond_fns)

        for attn, cross_attn,ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns)) + x
             x = cross_attn(x, context, attn_mask = attn_mask) + x
             x = ff(x, cond_fn = next(cond_fns)) + x
        return x
    
class TwoWayTransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.self_attn=TransformerAttention(dim,causal,dim_head,None,heads,False,attn_dropout)
        self.mlp=FeedForward(dim = dim, dropout = ff_dropout)
        self.state2token=TransformerAttention(dim,causal,dim_head,dim_context,heads,norm_context,attn_dropout)
        self.token2state=TransformerAttention(dim,causal,dim_head,dim_context,heads,norm_context,attn_dropout)

    def forward(
        self,
        token,
        state,
        mask = None,
        cond_fn: Optional[Callable] = None
    ):
        q,k=state,token
        q=self.self_attn(q)+q
        q=self.token2state(q,k)+q
        q=self.mlp(q)+q

        k=self.state2token(k,q)+k

        return q,k
    
class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TwoWayTransformerAttention(dim = dim, heads=heads,dim_context = dim,
                                                          norm_context=True,attn_dropout = attn_dropout,
                                                          ff_dropout = ff_dropout
                                                          ) 
                )
            
        self.final_attn=TransformerAttention(dim=dim,dim_head=dim_head,dim_context=dim,heads=heads,norm_context=True,dropout=attn_dropout)
        self.mlp=FeedForward(dim = dim, dropout = ff_dropout)

    def forward(
        self,
        token,
        state,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
       
        q=state
        k=token
        for layer in self.layers:
            q,k=layer(q,k) 
        
        x=self.final_attn(q,k)
        x=self.mlp(x)+x
        return x


class FilmEfficientNet(nn.Module):
    def __init__(self,model_path,output_dims=512):
        super().__init__()
        
        efficientnet = EfficientNet.from_pretrained(model_path)
        self.transform=transforms.Compose([
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.conv_stem=nn.Sequential(efficientnet._conv_stem,efficientnet._bn0)
        
        self.MBConvs=efficientnet._blocks
        self.cond_hidden_dims=[]
        for MBConv in self.MBConvs:
            self.cond_hidden_dims.append(MBConv._bn2.state_dict()['weight'].shape[0])
        
        self.mlp_head=nn.Linear(self.cond_hidden_dims[-1],output_dims)
    
    def forward(self,x,cond_fns=None):
        with torch.no_grad():
            x = self.transform(x)
            x = self.conv_stem(x)
            
            
        if not exists(cond_fns):
            cond_fns = (None,) * len(self.MBConvs)
        for block, cond_fn in zip(self.MBConvs, cond_fns):
            # with torch.no_grad():
            x = block(x)
            if exists(cond_fn):
                x = cond_fn(x)
                
        x = x.permute(0,2,3,1)
        x = self.mlp_head(x)
        x = x.permute(0,3,1,2)
        return x

class ClipEmbedder(nn.Module):
    def __init__(self,model_arch,model_path,clip_visual_dim=768,clip_text_dim=512,output_dims=512):
        super().__init__()
        import open_clip
        self.transform=transforms.Compose([
                        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),])
        self.model, _, preprocess = open_clip.create_model_and_transforms(model_arch, pretrained=model_path)
        self.tokenizer = open_clip.get_tokenizer(model_arch)
        self.img_mlp_head=nn.Linear(clip_visual_dim,output_dims)
        self.text_mlp_head=nn.Linear(clip_text_dim,output_dims)
    
    def forward(self,image,text,return_embed=False):
        device=image.device
        self.model.to(device)

        with torch.no_grad():
            image = self.transform(image)
            _, image_token = self.model.encode_image(image)
            text = self.tokenizer(text).to(device)
            text_token = self.model.encode_text(text)
        if return_embed:
            return image_token, text_token
        image_token = self.img_mlp_head(image_token)
        text_token = self.text_mlp_head(text_token)
        return image_token, text_token


# token learner module

class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x


class Tokenizer(nn.Module):
    def __init__(
        self,
        *,
        efficientnet_config,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_conditioner = True,
        conditioner_kwargs: dict = dict(),
        embed_dim=512
    ):
        super().__init__()
        self.efficientnet = FilmEfficientNet(**efficientnet_config)

        self.num_efficientnet_stages = len(self.efficientnet.cond_hidden_dims)

        if use_conditioner:
            conditioner_klass = TextConditioner

            self.conditioner = conditioner_klass(
                hidden_dims = (*tuple(self.efficientnet.cond_hidden_dims),),
                hiddens_channel_first = (*((True,) * self.num_efficientnet_stages), ),
                cond_drop_prob = cond_drop_prob,
                text_embed_stem_dim = embed_dim,
                **conditioner_kwargs
            )
        else:
            self.conditioner = None
        
        self.token_learner = TokenLearner(
            dim = embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.cond_drop_prob = cond_drop_prob

    @classifier_free_guidance
    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        cond_drop_prob = 0.
    ):
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[1], video.device
        cond_fns = self.conditioner(
            texts,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_efficientnet_stages), )
        )
        efficientnet_cond_fns = cond_fns[:]
        images, packed_shape = pack_one(video, '* c h w')

        tokens = self.efficientnet(
            images,
            cond_fns = efficientnet_cond_fns,
        )

        tokens = unpack_one(tokens, packed_shape, '* c h w')

        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
        return learned_tokens


@beartype
class KeyWorld(nn.Module):
    def __init__(
        self,
        *,
        # vit: MaxViT,
        clip_arch,
        clip_path,
        clip_visual_dim,
        clip_text_dim,
        state_num = 3,
        num_actions = 11,
        action_bins = 256,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict(),
        embed_dim=512,
        scene_transformer_depth=9,
        action_transformer_depth=3,
        dropout = 0.1,
    ):
        super().__init__()
        self.embedder = ClipEmbedder(model_arch=clip_arch,model_path=clip_path,clip_visual_dim=clip_visual_dim,clip_text_dim=clip_text_dim,output_dims=embed_dim)

        self.embed_dim = embed_dim

        self.scene_transformer = Transformer(
            dim = embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = scene_transformer_depth,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )
        self.action_transformer = Transformer(
            dim = embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = action_transformer_depth,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )
        self.cond_drop_prob = cond_drop_prob

        self.state_network=nn.Sequential(

            nn.Linear(state_num, 128),
            nn.Tanh(),
            nn.Linear(128, embed_dim),
            nn.Tanh(),
        )
        self.scene_net=nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, 768),
        )
        self.action_net=nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_actions*action_bins),
            Rearrange('... (a b) -> ... a b', b = action_bins)
        )
        self.predict_feature = None

    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        state = None,
        cond_drop_prob = 0.,
        return_embed = False,
        should_update=True
    ):
        assert state is not None
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        frames, device = video.shape[1], video.device
        images, packed_shape = pack_one(video, '* c h w')
        image_token, text_token = self.embedder(images,texts,return_embed=return_embed)
        n = image_token.shape[1]
        image_token = unpack_one(image_token, packed_shape, '* n c')

        if return_embed:
            return image_token,text_token
        learned_tokens = rearrange(image_token, 'b f n c -> b (f n) c')
        
        compared_feature = None
        if should_update:
            text_token = text_token.unsqueeze(1)
            prompt_token = torch.cat([learned_tokens,text_token], dim=1)

            predict_feature = self.scene_transformer(prompt_token, cond_fns = None)
            predict_feature = predict_feature[:,:frames*n] 
            predict_feature = rearrange(predict_feature, 'b (f n) d -> b f n d',f=frames,n=n)
            compared_feature = self.scene_net(predict_feature)
            predict_feature = reduce(predict_feature, 'b f n d -> b n d', 'mean',)
            self.predict_feature = predict_feature

        state_token=self.state_network(state)
        learned_tokens=torch.cat([learned_tokens,text_token,state_token,self.predict_feature], dim=1)
        
        attention_tokens = self.action_transformer(learned_tokens, cond_fns = None)
        
        pooled = reduce(attention_tokens, 'b fn d -> b d', 'mean')
        action = self.action_net(pooled)

        return action, compared_feature