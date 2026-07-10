import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MLP(nn.Module):
    def __init__(self, 
                 dim_list: List[int], 
                 activation: nn.Module = nn.LeakyReLU, 
                 use_batch_norm: bool = False, 
                 use_layer_norm: bool = False, 
                 last_activation: nn.Module = nn.Identity,
                 dropout: float = 0.1):
        super().__init__()
        n_last = len(dim_list) - 1
        self.layers = nn.ModuleList()
        for l, (in_dim, out_dim) in enumerate(zip(dim_list[:-1], dim_list[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            if l < n_last - 1:
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(out_dim))
                if use_layer_norm:
                    self.layers.append(nn.LayerNorm(out_dim))
                self.layers.append(activation())
                self.layers.append(nn.Dropout(dropout))
            else:
                self.layers.append(last_activation())    
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)

    

class Encoder(nn.Module):
    def __init__(self, 
                 dim_input: int, 
                 dim_output: int, 
                 dim_hidden: int, 
                 num_layers: int, 
                 use_batch_norm: bool = True, 
                 use_layer_norm: bool = False, 
                 dropout_rate: float = 0.1, 
                 activation: nn.Module = nn.LeakyReLU, 
                 variational: bool = False,
                 var_eps: float = 1e-4,
                 var_activation: nn.Module = None, 
                ):
        super().__init__()
        
        self.variational = variational
        self.var_eps = var_eps
        
        self.fc_layer = MLP(
            [dim_input, *([dim_hidden] * num_layers)],
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout=dropout_rate,
            last_activation=activation
        )
        self.mu_layer = nn.Linear(dim_hidden, dim_output)
        if variational:
            self.var_layer = nn.Linear(dim_hidden, dim_output)
            self.var_activation_fn = var_activation()
            
    def forward(self, x: Tensor):
        h = self.fc_layer(x)
        mu = self.mu_layer(h)
        if self.variational:
            var = self.var_activation_fn(self.var_layer(h) + self.var_eps)
            return mu, var
        return mu
    
            
class Decoder(nn.Module):
    def __init__(self, 
                 dim_input: int, 
                 dim_output: int, 
                 dim_hidden: int, 
                 num_layers: int, 
                 use_batch_norm: bool = True, 
                 use_layer_norm: bool = False, 
                 dropout_rate: float = 0.1, 
                 activation: nn.Module = nn.LeakyReLU,
                 output_act: str = 'linear'):
        super().__init__()
        self.output_act = output_act
        self.layers = MLP(
            [dim_input, *([dim_hidden] * num_layers), dim_output],
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout=dropout_rate,
            last_activation=nn.Identity
        )
        
    def forward(self, x):
        if self.output_act == 'linear':
            return self.layers(x)
        elif self.output_act == 'relu':
            return F.relu(self.layers(x))
        elif self.output_act == 'sigmoid':
            return torch.sigmoid(self.layers(x))
        elif self.output_act == 'softmax':
            return torch.softmax(self.layers(x), dim=-1)
        else:
            raise ValueError(f'Unknown output activation: {self.output_act}')
      
        
class ZINBDecoder(nn.Module):
    def __init__(self, 
                    dim_input: int, 
                    dim_output: int, 
                    dim_hidden: int, 
                    num_layers: int, 
                    use_batch_norm: bool = True, 
                    use_layer_norm: bool = False, 
                    dropout_rate: float = 0.1, 
                    activation: nn.Module = nn.LeakyReLU):
        super().__init__()
        self.layers = MLP(
            [dim_input, *([dim_hidden] * num_layers)],
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout=dropout_rate,
            last_activation=activation
        )
        self.meanscale_layer = nn.Sequential(
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=-1)
        )
        self.disp_layer = nn.Linear(dim_hidden, dim_output)
        self.pi_layer = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h = self.layers(x)
        meanscale = self.meanscale_layer(h)
        disp = torch.exp(self.disp_layer(h))
        pilogit = self.pi_layer(h)
        return meanscale, disp, pilogit


class NoVMappingMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, top_sparse: Optional[int | float] = None, temperature: float = 1.0,
                 use_v_proj: bool = False, use_out_ff: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.top_sparse = top_sparse
        assert self.head_dim * num_heads == embed_dim
        self.temperature = temperature
        self.use_v_proj = use_v_proj
        self.use_out_ff = use_out_ff
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        if use_v_proj:
            # project to embed_dim (not head_dim) so mean-over-heads is well-defined
            self.v_proj = nn.Linear(embed_dim, embed_dim)

        if use_out_ff:
            self.out_proj = nn.Sequential(
                MeltingMHAOutput(),
                nn.Linear(embed_dim * num_heads, embed_dim),
            )
        # when no out_ff, output is averaged over heads to keep embed_dim shape

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        Lq, Lk = q.size(0), k.size(0)
        q = self.q_proj(q).reshape(Lq, self.num_heads, self.head_dim).transpose(0, 1) # [H, Lq, d]
        k = self.k_proj(k).reshape(Lk, self.num_heads, self.head_dim).transpose(0, 1) # [H, Lk, d]
        if self.use_v_proj:
            # v_proj maps to embed_dim; broadcast across heads without splitting
            v = self.v_proj(v).unsqueeze(0).expand(self.num_heads, -1, -1)  # [H, Lk, embed_dim]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5 * self.temperature) # [H, Lq, Lk]
        if self.top_sparse is not None:
            if isinstance(self.top_sparse, int) and (0 < self.top_sparse <= Lk):
                topk = self.top_sparse
            elif isinstance(self.top_sparse, float) and (0 < self.top_sparse <= 1):
                topk = int(self.top_sparse * Lk)
            else:
                raise ValueError(f"Invalid top_sparse: {self.top_sparse}")

            _, topk_indices = torch.topk(attn_scores, topk, dim=-1) # [H, Lq, topk]
            mask = torch.zeros_like(attn_scores).scatter_(-1, topk_indices, 1) # [H, Lq, Lk]
            attn_scores = attn_scores.masked_fill(mask == 0, -torch.inf) # [H, Lq, Lk]

        attn_weights = torch.softmax(attn_scores, dim=-1) # [H, Lq, Lk]
        out = attn_weights @ v  # [H, Lq, d] if use_v_proj else [H, Lq, embed_dim]
        if self.use_out_ff:
            return self.out_proj(out), attn_weights  # [H, Lq, Lk]
        else:
            return out.mean(dim=0), attn_weights  # [H, Lq, Lk]
    
    
class MinMaxScaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return (x - x.min()) / (x.max() - x.min())
    

class MeltingMHAOutput(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.transpose(0, 1).reshape(x.size(1), -1)
