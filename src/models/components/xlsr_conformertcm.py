import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
import fairseq

import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones
from src.models.components.conformer import ConformerBlock

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1, pooling='mean', type='conv', **kwargs):
        super(MyConformer, self).__init__()
        self.pooling=pooling
        self.dim_head=int(emb_size/heads)
        self.dim=emb_size
        self.heads=heads
        self.kernel_size=kernel_size
        self.n_encoders=n_encoders
        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
        self.conv_dropout = kwargs.get('conv_dropout', 0.0)
        self.ff_dropout = kwargs.get('ff_dropout', 0.0)
        self.attn_dropout = kwargs.get('attn_dropout', 0.0)
        self.encoder_blocks=_get_clones(ConformerBlock(dim = emb_size, dim_head=self.dim_head, heads= heads, 
                ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size, type=type, 
                conv_dropout = self.conv_dropout, ff_dropout = self.ff_dropout, attn_dropout = self.attn_dropout
                ),
            n_encoders)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)
    
    def forward(self, x): # x shape [bs, tiempo, frecuencia]
        x = x + self.positional_emb[:, :x.size(1), :]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
        list_attn_weight = []
        for layer in self.encoder_blocks:
            x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
        if self.pooling=='mean':
            embedding = x.mean(dim=1)
        elif self.pooling=='max':
            embedding = x.max(dim=1)[0]
        else:
            # first token
            embedding=x[:,0,:] #[bs, emb_size]
        out=self.fc5(embedding) #[bs,2]
        return out, embedding
    

class SSLModel(nn.Module):
    def __init__(self, ssl_pretrained_path):
        super(SSLModel, self).__init__()
        cp_path = ssl_pretrained_path   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        #self.out_dim = 1024
        self.out_dim = self.model.cfg.encoder_embed_dim

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

    def forward(self, input_data):
        return self.extract_feat(input_data)


class Model(nn.Module):
    def __init__(self, args, ssl_pretrained_path):
        super().__init__()
        self.front_end = SSLModel(ssl_pretrained_path)
        self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
 
    def forward(self, x, last_emb=False):
        x.requires_grad = True
        x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score =self.backend(x)
        if last_emb:
            return attn_score
        return out
