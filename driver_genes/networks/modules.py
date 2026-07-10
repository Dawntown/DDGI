from torch import Tensor
from typing import Any, Optional, Tuple, Dict
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from .layers import MLP, Encoder, Decoder, ZINBDecoder, NoVMappingMultiHeadAttention
from ..losses import *
from ..myutils import Args, is_verbose
from ..data import load_prior_data, load_batchs


act_dict = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'elu': nn.ELU,
    'prelu': nn.PReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'swish': nn.SiLU,
    'softplus': nn.Softplus,
    'identity': nn.Identity,
}


class BaseExtractor(nn.Module, ABC):
    """Base class for perturbation salience extractors.
    
    Provides common functionality for encoder-decoder architectures with optional batch layers.
    Subclasses should implement the counterfactual control computation in _compute_counterfactual.
    """
    def __init__(self,
                 dim_genes: int,
                 dim_latent: int,
                 dim_hidden: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 act: str = 'leakyrelu',
                 num_batchs: int = 1,
                 variational: bool = False,
                 rec_loss: str = 'mse',
                 beta: float = 0.1,
                 **kwargs):
        super().__init__()
        self.dim_genes = dim_genes
        self.dim_latent = dim_latent
        self.num_batchs = num_batchs
        self.variational = variational
        self.beta = beta
        self.coupling = None # alignment matrix between source and target batches [N_y, N_x]: normalized to sum over each row to 1
        self.store_coupling = False  # set True at inference time to include coupling in results dict
        
        # Batch layer
        if num_batchs > 1:
            if is_verbose():
                print(f"Using batch layer with {num_batchs} batches")
            self.batch_layer = nn.Embedding(
                num_batchs,
                dim_latent,
                padding_idx=0
            )
        else:
            self.batch_layer = None
        
        # Encoder
        encoder_input_dim = dim_genes if self.batch_layer is None else (dim_genes + dim_latent)
        self.encoder = Encoder(
            dim_input=encoder_input_dim,
            dim_output=dim_latent,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout,
            activation=act_dict[act],
            variational=variational,
            var_activation=nn.Identity if variational else None, # for logvar, use identity activation
        )
        
        # Decoder
        if rec_loss == 'zinb':
            self.decoder = ZINBDecoder(
                dim_input=dim_latent,
                dim_output=dim_genes,
                dim_hidden=dim_hidden,
                num_layers=num_layers,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout,
                activation=act_dict[act],
            )
        else:
            self.decoder = Decoder(
                dim_input=dim_latent,
                dim_output=dim_genes,
                dim_hidden=dim_hidden,
                num_layers=num_layers,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout,
                activation=act_dict[act],
                output_act='relu'
            )
        
        if kwargs and is_verbose():
            print(f"Unused kwargs: {kwargs}")

    
    def _stabilize_input(self, expr: Tensor):
        """Stabilize input expression data for ZINB decoder."""
        if isinstance(self.decoder, ZINBDecoder):
            return torch.log1p(expr), torch.sum(expr, dim=1, keepdim=True)
        else:
            return expr, None
        
    def _apply_batch_layer(self, expr: Tensor, batch: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Apply batch layer if available."""
        if self.batch_layer is not None and batch is not None:
            batch_emb = self.batch_layer(batch)
            expr = torch.cat([expr, batch_emb], dim=1)
            return expr, batch_emb
        return expr, 0
    
    def _encode(self, expr: Tensor, batch: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode input cells.
        
        For VAE: applies reparameterization trick to sample z from N(mu, var).
        Returns latent representations and optionally variances (NOT perturbation effects).
        
        Note: Perturbation effects (sy) should be computed by _compute_counterfactual, not here.
        """
        expr, batch_emb = self._apply_batch_layer(expr, batch)
        if self.variational:
            mu, logvar = self.encoder(expr)
            return mu, logvar, batch_emb
        else:
            mu = self.encoder(expr)
            return mu, None, batch_emb
        
    def _sample_z(self, mu: Tensor, logvar: Tensor = None) -> Tensor:
        """Sample z from N(mu, exp(logvar))."""
        if logvar is None:
            return mu
        else:
            eps = torch.randn_like(mu)
            return mu + torch.exp(logvar * 0.5) * eps
    
    def _decode(self, z: Tensor, batch_emb: Tensor = 0, total_counts: Tensor = None):
        """Decode latent representation."""
        if isinstance(self.decoder, ZINBDecoder):
            prop, disp, pilogit = self.decoder(z + batch_emb)
            expr_hat = prop * total_counts
            disp = torch.clamp(disp, min=1e-4)
            return expr_hat, (expr_hat, disp, pilogit)
        else:
            return self.decoder(z + batch_emb), None
    
    @abstractmethod
    def _decompose(self, zy: Tensor, zx: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute counterfactual control representation.
        
        Subclasses should implement this method.
        """
        raise NotImplementedError
    
    def forward(self, inputs: dict, **kwargs) -> Dict[str, Tensor]:
        """Forward pass.
        
        Args:
            inputs: Dictionary containing 'x', 'y', optionally 'x_batch', 'y_batch'
            **kwargs: Additional arguments (e.g., ae_only)
            
        Returns:
            Dictionary with 'x_hat', 'y_hat', 'zx', 'zxy', 'zy', 'sy', 'dy' (depending on whether coupling is used)
        """        
        # Apply batch layer if provided (for batch correction)
        x, x_total = self._stabilize_input(expr=inputs['x'])
        x_batch = inputs.get('x_batch', None)
        mu_zx, logvar_zx, batch_emb_x = self._encode(x, batch=x_batch)
        smp_zx = self._sample_z(mu_zx, logvar_zx)
        x_hat, x_params = self._decode(smp_zx, batch_emb_x, x_total)
        
        results = {
            'x_hat': x_hat,
            'zx': mu_zx,
        }
        
        if kwargs.get('ae_only', False):
            if kwargs.get('loss', False):
                l_rec = self.get_l_rec(inputs['x'], x_params)
                # only assume control cells are drawing from Normal distribution
                l_kld = self.get_l_kld(mu_zx, logvar_zx)
                return results, {'vae': l_rec + l_kld * self.beta, 'rec': l_rec, 'kld': l_kld}
            else:
                return results
        
        y, y_total = self._stabilize_input(expr=inputs['y'])
        y_batch = inputs.get('y_batch', None)
        mu_zy, logvar_zy, batch_emb_y = self._encode(y, batch=y_batch)
        zxy, sy = self._decompose(mu_zy, mu_zx, **kwargs)
        smp_zxy = self._sample_z(zxy, logvar_zy)
        y_hat, y_params = self._decode(smp_zxy + sy, batch_emb_y, y_total)
        
        results.update({
            'y_hat': y_hat,
            'zxy': zxy,
            'zy': mu_zy,
            'sy': sy
        })
        
        if isinstance(self.coupling, Tensor):
            coupling_for_dy = self.coupling.mean(dim=0) if self.coupling.dim() == 3 else self.coupling
            results['dy'] = y - coupling_for_dy @ x
            if self.store_coupling:
                results['coupling'] = self.coupling.detach().cpu()  # [H, B, B]
            self.coupling = None
        else:
            results['dy'] = y - x
        
        if kwargs.get('loss', False):
            l_rec = self.get_l_rec(inputs['x'], x_hat, x_params) + self.get_l_rec(inputs['y'], y_hat, y_params)
            l_kld = 2 * self.get_l_kld(mu_zx, logvar_zx) # only assume control cells are drawing from Normal distribution
            return results, {'vae': l_rec + l_kld * self.beta, 'rec': l_rec, 'kld': l_kld}
        else:
            return results
        
    def get_l_rec(self, expr: Tensor, expr_hat: Tensor = None, params: Tuple = None, **kwargs):
        if isinstance(self.decoder, ZINBDecoder):
            return zinb(expr, params)
        else:
            return mse(expr, expr_hat)
        
    def get_l_kld(self, mu: Tensor, logvar: Tensor = None, **kwargs):
        if self.variational and logvar is not None:
            return kl_div(mu, logvar)
        else:
            return torch.tensor(0.0, device=mu.device)


class PertVAExtractorCrossAttn(BaseExtractor):
    """Cross-attention based perturbation salience extractor (attn model).
    
    Uses multi-head attention to compute counterfactual control representation.
    This is the primary extractor model for driver gene identification.
    """
    def __init__(self,
                 dim_latent: int,
                 num_heads: int = 1,
                 top_sparse: Optional[int | float] = None,
                 temperature: float = 1.0,
                 use_v_proj: bool = False,
                 use_out_ff: bool = True,
                 **kwargs):
        super().__init__(dim_latent=dim_latent, **kwargs)

        self.attn = NoVMappingMultiHeadAttention(
            embed_dim=dim_latent,
            num_heads=num_heads,
            top_sparse=top_sparse,
            temperature=temperature,
            use_v_proj=use_v_proj,
            use_out_ff=use_out_ff,
        )
        if is_verbose():
            print(f"Using {self.__class__.__name__}")
        
    def _decompose(self, zy: Tensor, zx: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute counterfactual control using cross-attention."""
        zxy, coupling = self.attn(zy, zx, zx)
        # zxy = (coupling.mean(dim=0) if coupling.dim() == 3 else coupling) @ zx
        self.coupling = coupling
        return zxy, zy - zxy
        

class PriorEncoder(nn.Module):
    def __init__(self, 
                 embeddings: Any = None,
                 finetune: bool = False,
                 num_genes: int = None,
                 dim_hidden: int = None
                ):
        super().__init__()
        self._finetune = finetune
        if embeddings is None:
            self.embed_layer = nn.Embedding(num_genes, dim_hidden)
        elif isinstance(embeddings, torch.Tensor):
            self.embed_layer = nn.Sequential(
                nn.Embedding.from_pretrained(embeddings, freeze=not finetune),
                # nn.Linear(embeddings.size(1), dim_hidden),
                MLP([embeddings.size(1), dim_hidden, dim_hidden], activation=nn.LeakyReLU),
            )
        else:
            raise TypeError(f"Unsupported prior data type: {type(embeddings)}")
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embed_layer(x)


class PriorFuser(nn.Module):
    def __init__(self, 
                 strategy: str,
                 dim_latent: int,
                 dim_feat: int, 
                 partition_size: Optional[int] = 32,
                ):
        super().__init__()
        if strategy != 'bilinear':
            raise ValueError("DDGI only supports fuser.strategy='bilinear'.")
        self.strategy = strategy
        self.dim_latent = dim_latent
        self.dim_feat = dim_feat
        self.partition_size = partition_size
        self.fuse_mlp = nn.Bilinear(
            dim_latent, dim_latent, dim_feat
        )
            
    def forward(self, s: Tensor, k: Tensor) -> Tensor:
        """
        Input:
            s: [N, dim_latent]
            k: [G, dim_latent]
        Output:
            [N, G, dim_feat]
        """
        s = s.unsqueeze(1) # [N, 1, dim_latent]
        k = k.unsqueeze(0) # [1, G, dim_latent]
        p = []
        for i in range(np.ceil(k.size(1) / self.partition_size).astype(int)):
            start = i * self.partition_size
            end = min(start + self.partition_size, k.size(1))
            p.append(self.fuse_mlp(
                s.expand(-1,end-start,-1),
                k[:,start:end,:].expand(s.size(0),-1,-1)
            ))
        return torch.cat(p, dim=1)


class DriverGeneFinder(nn.Module):
    def __init__(self, 
                 extractor_args: Args = 'None',
                 fuser_args: Args = None,
                 pe_type: str = 'none',
                 organism: str = 'human',
                 finetune: bool = False,
                 output_dir: str = None,
                 rec_loss: str = 'mse', 
                 dis_loss: str = 'edist_euclidean',
                 cls_loss: str = 'binary_cross_entropy',
                 extractor_type: str = 'attn',
                 with_batch_layer: bool = False,
                 **kwargs
                 ):
        super().__init__()
        
        self.extractor_type = extractor_type
        extractor_args.rec_loss = rec_loss
        extractor_args.dis_loss = dis_loss
        self.with_batch_layer = with_batch_layer
        if with_batch_layer:
            num_batchs = len(load_batchs(output_dir))
            extractor_args.num_batchs = num_batchs
            
        # initialize extractor
        if extractor_type != 'attn':
            raise ValueError("DDGI only supports identifier.extractor_type='attn'.")
        self.extractor = PertVAExtractorCrossAttn(**extractor_args.getall())
            
        # initialize fuser
        self.fuser = PriorFuser(**fuser_args.getall())
        
        # load prior data
        self.candidate_genes, self.prior_data = load_prior_data(
            output_dir, 
            pe_type, 
            organism
        )
        
        # NOTE: use the parameter setting in object instead of args
        # to leverage the default setting of class   
        if self.prior_data is None:
            self.node_embed = PriorEncoder(
                num_genes=len(self.candidate_genes),
                dim_hidden=self.extractor.dim_latent,
            )
        elif isinstance(self.prior_data, Tensor):
            self.node_embed = PriorEncoder(
                embeddings=self.prior_data, 
                finetune=finetune,
                dim_hidden=self.extractor.dim_latent,
            )
        else:
            self.node_embed = PriorEncoder(
                embeddings=self.prior_data, 
                dim_hidden=self.extractor.dim_latent,
            )

        if self.fuser.dim_feat > 1:
            self.identifier = nn.Sequential(
                MLP([self.fuser.dim_feat, self.fuser.dim_feat, 1]),
                nn.Flatten(start_dim=1),
            )        
        else:
            self.identifier = nn.Flatten(start_dim=1)
            
        self.cls_loss = globals()[cls_loss]

        if kwargs and is_verbose():
            print(f"Unused kwargs: {kwargs}")

        if is_verbose():
            print(self)

    def forward(self, 
                inputs: dict,
                **kwargs,
        ) -> Dict[str, Tensor]:
        """
        Input:
            x: [N, g]
            y: [N, g]
            is_sup: bool, whether the input is supervised or not
        """
        
        results = self.extractor(inputs, **kwargs)
        if kwargs.get('ae_only', False):
            return results
        
        if kwargs.get('loss', False):
            # unpack results and losses
            results, losses = results
            
        k = self.node_embed(torch.arange(len(self.candidate_genes), device=inputs['x'].device))
        clslogits = self.identifier(self.fuser(results['sy'], k))
        results['cls'] = clslogits.sigmoid()

        if kwargs.get('loss', False):
            losses['cls'] = self.get_l_cls(inputs['psi'], clslogits, **kwargs)
            return results, losses
        
        return results
    
    def get_l_cls(self, psi: Tensor, clslogits: Tensor, **kwargs):
        return self.cls_loss(clslogits, psi, **kwargs)
            
            
