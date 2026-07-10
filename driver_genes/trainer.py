import os, warnings, yaml
from collections import deque
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from .myutils import (
    Args, load_yaml,
    dict_to_args, args_to_dict,
    paste_paths, reduce_dict_list,
    is_verbose,
)
from .data import prepare_datasets, concat_results
from .metrics import fetch_metrics
from .networks import DriverGeneFinder
from .losses import *




class PLData(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 adata_file: str | list[str], 
                 split_file: str | list[str], 
                 sample_size: int = 32,
                 batch_size: int = 4,
                 num_workers: int = 8,
                 output_dir: str = None,
                 layer_name: str = 'X',
                 var_subset: str = 'highly_variable',
                 cotrain_mode: bool = False,
                 sample_neg: bool = False,
                 batch_key: str = None,
                 condition_key: str = 'perturbation',
                 source_cell: str = 'control',
                 resampling_mode: bool = True,
                 ):
        super().__init__()
        if is_verbose():
            print("Preparing data...")
        if (batch_key is not None and batch_key != 'None') and not cotrain_mode:
            if is_verbose():
                print(f"Batch layer ({batch_key}) can only be used in cotrain mode, but cotrain_mode is False, batch key will be ignored")
            warnings.warn(f"Batch layer ({batch_key}) can only be used in cotrain mode, but cotrain_mode is False, batch key will be ignored")
            batch_key = None

        (
            self.ds_dict, 
            self.pertgenes, 
            self.split_df, 
            self.gene_order, 
            self.organism,
            self.batchs,
        ) = prepare_datasets(
            adata_path=paste_paths(data_dir, adata_file),
            split_path=paste_paths(data_dir, split_file),
            layer_name=layer_name,
            var_subset=var_subset,
            sample_size=sample_size,
            cotrain_mode=cotrain_mode,
            sample_neg=sample_neg,
            batch_key=batch_key,
            condition_key=condition_key,
            source_cell=source_cell,
            resampling_mode=resampling_mode,
        )
        
        if is_verbose():
            print(f"ds_dict:")
            for split, ds_dict_split in self.ds_dict.items():
                print(f"\t{split}: {list(ds_dict_split.keys())}")
                        
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cotrain_mode = cotrain_mode
        self.sample_neg = sample_neg
        self.dump_auxiliary(output_dir)
                    
        
    def dump_auxiliary(self, output_dir):
        """Dump auxiliary data (split_df and pertgenes) for training and backup."""
        self.split_df.to_csv(os.path.join(output_dir, 'split_df.csv.gz'), index=False)
        np.savetxt(
            os.path.join(output_dir, 'pertgenes.txt'), 
            self.pertgenes, fmt='%s'
        )
        if self.batchs is not None:
            np.savetxt(
                os.path.join(output_dir, 'batchs.txt'), 
                self.batchs, fmt='%s'
            )
            if is_verbose():
                print(f"Dumping split_df, pertgenes, and batchs to {output_dir}")
        else:
            if is_verbose():
                print(f"Dumping split_df and pertgenes to {output_dir}")
        
    def train_dataloader(self):
        self.initialize(dataset_name='train')
        return CombinedLoader({ad_name: DataLoader(
            ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        ) for ad_name, ds in self.ds_dict['train'].items()}, mode='min_size')
    
    def val_dataloader(self):
        self.initialize(dataset_name='val')
        return CombinedLoader({ad_name: DataLoader(
            ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        ) for ad_name, ds in self.ds_dict['val'].items()}, mode='min_size')
    
    def test_dataloader(self, seed=None):
        if seed is not None:
            self.initialize(dataset_name='test', seed=seed)
        return CombinedLoader({ad_name: DataLoader(
            ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False, 
        ) for ad_name, ds in self.ds_dict['test'].items()}, mode='min_size')
    
    def initialize(self, dataset_name: str, seed=None):
        for ds in self.ds_dict[dataset_name].values():
            ds.initialize(seed=seed)



class PLModel(LightningModule):
    def __init__(self, 
                 extractor_args: Args, 
                 fuser_args: Args, 
                 identifier_args: Args,
                 trainer_args: Args,
                 ):
        super().__init__()
        if is_verbose():
            print("Preparing model...")
        # self.save_hyperparameters()
        
        self.model = DriverGeneFinder(
            extractor_args=extractor_args,
            fuser_args=fuser_args, 
            **identifier_args.getall()
        )
        
        self.cotrain_weight = trainer_args.get('cotrain_weight', 1.0)
        # moving average monitor config
        self.monitor_metric = trainer_args.get('monitor_metric', 'Lmix')
        self.monitor_ma_window = trainer_args.get('monitor_ma_window', 1)
        self._val_metric_history = None
        if isinstance(self.monitor_ma_window, int) and self.monitor_ma_window > 1:
            self._val_metric_history = deque(maxlen=self.monitor_ma_window)
            self._epoch_val_metric_values = []
        
        # setup metrics
        self.train_metrics_func = fetch_metrics(
            metric_key_list=trainer_args.get('train_metrics', 'core'), 
            ignore_const=True
        )
        self.val_metrics_func = fetch_metrics(
            metric_key_list=trainer_args.get('train_metrics', 'core'), 
            ignore_const=True
        )
        self.test_metrics_func = fetch_metrics(
            metric_key_list=trainer_args.get('test_metrics', 'all'), 
            ignore_const=True
        )
        
        # setup losses
        loss_weights = trainer_args.get('loss_weights', {
            'vae': 1.0,
            'dis': 10.0,
            'cls': 100.0
        })
        
        assert hasattr(loss_weights, 'vae'), f"loss_weights must contain 'vae' key, but got {args_to_dict(loss_weights)}"

        if trainer_args.get('mixing_strategy', 'predefined') == 'uncertainty':
            self.mixing_losses = UncertaintyWeighting(init_weights=loss_weights)
        # elif trainer_args.get('mixing_strategy', 'predefined') == 'adaptive':
        #     self.mixing_losses = AdaptiveWeightLoss()
        elif trainer_args.get('mixing_strategy', 'predefined') == 'predefined':
            self.mixing_losses = PredefinedWeightLoss(weights=loss_weights)
        else:
            raise ValueError(f"Invalid mixing strategy: {trainer_args.get('mixing_strategy', None)}")

        self.learning_rate = trainer_args.get('learning_rate', 0.001)
        self.selected_metrics = trainer_args.get('bar_metrics', ['Lmix', 'label#AUROC'])
        self.test_step_outputs = [] # for on_test_step_end
        self.predict_step_outputs = [] # for on_predict_step_end
        self.predictions = None
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def _process_batch(self, batch, metrics_func, prefix='', ae_only_allowed=False, return_results=False):
        """Process a batch and compute losses and metrics.
        
        Args:
            batch: Batch data (dict or CombinedLoader dict/tuple)
            metrics_func: Metrics function to use
            prefix: Prefix for logging (e.g., 'Tr', 'Va', 'Te')
            ae_only_allowed: Whether cotrain (ae_only) batches are allowed
            return_results: Whether to return results dict for saving
            
        Returns:
            Tuple of (total_loss, metrics_all, results_dict)
        """

        losses_sub_list = []
        metrics_sub_list = []
        results_dict = {}
        
        for ad_name, batch_sub in batch.items():
            ae_only = ad_name.endswith('_cotrain') if ae_only_allowed else False
            
            if ae_only:
                # Remove psi for cotrain mode to avoid label leakage
                batch_sub = {k: v for k, v in batch_sub.items() if k != 'psi'}
            
            results_sub, losses_sub = self.model(batch_sub, ae_only=ae_only, loss=True)
            
            # Apply cotrain weight if applicable
            if ae_only:
                losses_sub = {k: (losses_sub[k] * self.cotrain_weight) for k in losses_sub.keys()}
            
            losses_sub_list.append(losses_sub)
            
            # Skip metrics for ae_only batches or batches with no positive labels
            if ae_only or (batch_sub.get('psi') is not None and batch_sub['psi'].sum() == 0):
                if return_results:
                    results_sub.update(batch_sub)
                    results_dict[ad_name] = results_sub
                continue
            
            # Compute metrics
            metrics_sub = metrics_func(results_sub['cls'], batch_sub['psi'])
            metrics_sub.update({f'L{k}': v.detach() for k, v in losses_sub.items()})
            metrics_sub_list.append(metrics_sub)
            
            # Log per-dataset metrics (only if multiple datasets)
            if len(batch) > 1:
                for k, v in metrics_sub.items():
                    self.log(
                        f"{k}/{prefix}/{ad_name}", v,
                        batch_size=batch_sub['x'].shape[0],
                        on_step=(prefix == 'Tr'),
                        on_epoch=(prefix == 'Va' or prefix == 'Te'),
                        logger=True
                    )
            
            if return_results:
                results_sub.update(batch_sub)
                results_dict[ad_name] = results_sub
        
        # Aggregate losses and metrics
        losses_all = reduce_dict_list(losses_sub_list, method='mean')
        metrics_all = reduce_dict_list(metrics_sub_list, method='mean')
        total_loss, loss_dict_all = self.mixing_losses(losses_all)
        metrics_all.update({f'L{k}': v.detach() for k, v in loss_dict_all.items()})
        
        # Log aggregated metrics
        for k, v in metrics_all.items():
            show_in_progbar = k in self.selected_metrics
            self.log(
                f"{k}/{prefix}/a", v,
                on_step=(prefix == 'Tr'),
                on_epoch=(prefix == 'Va' or prefix == 'Te'),
                prog_bar=show_in_progbar,
                logger=True
            )
        
        if return_results:
            return total_loss, metrics_all, results_dict
        return total_loss, metrics_all

    def training_step(self, batch, batch_idx):
        total_loss, _ = self._process_batch(
            batch, 
            self.train_metrics_func, 
            prefix='Tr', 
            ae_only_allowed=True
        )
        return total_loss
    
    def on_train_epoch_end(self):
        self.train_metrics_func.reset()
                
    def validation_step(self, batch, batch_idx):
        total_loss, metrics_all = self._process_batch(
            batch, 
            self.val_metrics_func, 
            prefix='Va'
        )
        # Capture current-step value for moving average
        if self._val_metric_history is not None:
            try:
                if self.monitor_metric in metrics_all:
                    self._epoch_val_metric_values.append(float(metrics_all[self.monitor_metric]))
            except Exception:
                pass
        return total_loss
    
    def on_validation_epoch_start(self):
        if hasattr(self, '_epoch_val_metric_values'):
            self._epoch_val_metric_values = []
    
    def on_validation_epoch_end(self):
        # compute and log moving-average validation metric if enabled
        if self._val_metric_history is not None and self.trainer is not None:
            target_key = f"{self.monitor_metric}/Va/a"
            current_val = self.trainer.callback_metrics.get(target_key, None)
            try:
                if current_val is not None:
                    current_scalar = float(current_val)
                elif len(self._epoch_val_metric_values) > 0:
                    current_scalar = float(np.mean(self._epoch_val_metric_values))
                else:
                    current_scalar = None
                if current_scalar is not None:
                    self._val_metric_history.append(current_scalar)
                    ma_value = float(np.mean(list(self._val_metric_history)))
                    ma_key = f"{self.monitor_metric}_MA{self.monitor_ma_window}/Va/a"
                    # log as epoch metric so callbacks can monitor it
                    self.log(ma_key, ma_value, on_epoch=True, prog_bar=False, logger=True)
            except Exception:
                pass
        self.val_metrics_func.reset()
    
    def test_step(self, batch, batch_idx):
        total_loss, _, results_dict = self._process_batch(
            batch, 
            self.test_metrics_func, 
            prefix='Te',
            return_results=True
        )
        # Move results to CPU to avoid OOM
        results_dict = strip_results_from_cuda(results_dict)
        self.test_step_outputs.append(results_dict)
        return total_loss
    
    def on_test_epoch_end(self):
        """
        Input:
            outputs: list[dict(ad_name: dict(metric_name: metric_value))]
        Output:
            ad_results: dict(ad_name: dict(metric_name: tensor(list[metric_value])))
        """
        if is_verbose():
            print(f"Concatenating test step outputs: {len(self.test_step_outputs)}")
        self.predictions = concat_results(self.test_step_outputs)
        # Ensure final predictions are on CPU to avoid OOM
        self.predictions = strip_results_from_cuda(self.predictions)
        self.test_step_outputs.clear()
        self.test_metrics_func.reset()
        
    def on_test_start(self):
        self.test_step_outputs = []
        self.predictions = None
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: should we expect the batch in predict step as a single adata?
        # if so, we should remove the (ad_name, batch_sub) loop.
        # WARNING: predict step only supports sequential dataloader.
        results_dict = {}
        ad_name_idx = dataloader_idx
        batch_sub = batch
        results_sub = self.model(batch_sub)
        results_sub.update(batch_sub)
        results_dict.update({ad_name_idx: results_sub})
        # Move results to CPU to avoid OOM
        results_dict = strip_results_from_cuda(results_dict)
        self.predict_step_outputs.append(results_dict)
        return results_dict
    
    def on_predict_epoch_end(self):
        if is_verbose():
            print(f"Concatenating predict step outputs: {len(self.predict_step_outputs)}")
        self.predictions = concat_results(self.predict_step_outputs)
        # Ensure final predictions are on CPU to avoid OOM
        self.predictions = strip_results_from_cuda(self.predictions)
        self.predict_step_outputs.clear()
        
    def on_predict_start(self):
        self.predict_step_outputs = []
        self.predictions = None
    
    
    
def setup_trainer(output_dir,
                  max_epochs=100,
                  device='cpu', 
                  monitor_metric='Lmix', 
                  monitor_mode='min', 
                  monitor_ma_window=5,
                  debugging=False,
                  **kwargs):
    
    if is_verbose():
        print("Setting up trainer...")
    torch.set_float32_matmul_precision(kwargs.get('precision', 'high'))
            
    callback_list = []
    if debugging:
        callback_list.append(ModelSummary(max_depth=2))
        profiler = PyTorchProfiler(profile_memory=True, record_shapes=True)
        loggers = TensorBoardLogger(save_dir=output_dir, name='logs')

    else:
        # decide which metric to monitor: moving-average if window > 1
        suffix = '/Va/a'
        monitor_key = f"{monitor_metric}{suffix}" if not (isinstance(monitor_ma_window, int) and monitor_ma_window > 1) else f"{monitor_metric}_MA{monitor_ma_window}{suffix}"
        callback_list.append(ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename="model-{epoch:02d}",
            save_top_k=1, # only save the best model to reduce disk usage
            verbose=False,
            monitor=monitor_key,
            mode=monitor_mode,
            save_last=True,
        ))
        profiler = None
        loggers = TensorBoardLogger(save_dir=output_dir, name='logs')
        
        if kwargs.get('early_stopping', False):
            callback_list.append(
                EarlyStopping(
                    monitor=monitor_key,
                    patience=kwargs.get('patience', 10),
                    verbose=False,
                    mode=monitor_mode
                ))
        
    trainer = Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        callbacks=callback_list,
        logger=loggers,
        accelerator='gpu' if device != 'cpu' else 'cpu',
        devices=1,
        log_every_n_steps=10,
        profiler=profiler,
        reload_dataloaders_every_n_epochs=1,
        precision=None if kwargs.get('precision', 'high') == 'high' else '16-mixed',
    )
    
    return trainer


def prepare_config(*, 
                   config_path: str | None = None, 
                   config_dict: dict | None = None, 
                   rerun: bool = False,
                   **kwargs):
    
    if config_path is not None:
        config = load_yaml(config_path)
    elif config_dict is not None:
        config = config_dict
    else:
        raise ValueError("Either config_path or config_dict must be provided")
    
    args = dict_to_args(config)

    args.identifier.pe_type = args.identifier.get('pe_type', 'none')
    args.identifier.extractor_type = args.identifier.get('extractor_type', 'attn')
    if not hasattr(args, 'fuser'):
        args.fuser = dict_to_args({})
    args.fuser.strategy = args.fuser.get('strategy', 'bilinear')
    args.fuser.dim_feat = args.fuser.get('dim_feat', 1)
    
    if kwargs.get('print_info', False):
        info_args = args.info.getall()
        for k, v in info_args.items():
            if is_verbose():
                print(f"{k.upper()}: {v}")
            
    # set seed
    if hasattr(args.trainer, 'seed'):
        torch.manual_seed(args.trainer.seed)
        torch.cuda.manual_seed(args.trainer.seed)
        np.random.seed(args.trainer.seed)
        if is_verbose():
            print(f"Setting seed to {args.trainer.seed}")

    if rerun:
        return args
    else:
        if hasattr(args.trainer, 'seed'):
            args.trainer.version = f'{args.trainer.version}-seed{args.trainer.seed}'       
            
        # config dim_latent of extractor and dim_hidden of fuser
        args.fuser.dim_latent = args.extractor.dim_latent
        # trainer should dump logs, checkpoints, etc. in the output_dir
        args.trainer.output_dir = os.path.join(
            args.trainer.output_dir, 
            args.trainer.job_name, 
            args.trainer.version
        )
        # dataset should dump the perturbed gene list in the output_dir
        args.dataset.output_dir = args.trainer.output_dir
        # model should load the perturbed gene list from the output_dir
        args.identifier.output_dir = args.trainer.output_dir
        
        os.makedirs(args.trainer.output_dir, exist_ok=True)
        
        # save the config to the output_dir for reproducibility
        yaml.dump(
            args_to_dict(args), 
            open(f'{args.trainer.output_dir}/config.yaml', 'w'), 
            default_flow_style=False
        )
        
        return args


def build_predict_loader(ds_dict: dict, batch_size: int, num_workers: int):
    return CombinedLoader({ad_name: DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False, 
    ) for ad_name, ds in ds_dict.items()}, mode='sequential')
    
    
def setup_data_module(args: Args):
    pl_data_module = PLData(**args.dataset.getall())
    args.extractor.dim_genes = len(pl_data_module.gene_order)
    return pl_data_module

def setup_model_module(args: Args, state_dict_path: str = None, checkpoint_path: str = None):
    if checkpoint_path is not None:
        if is_verbose():
            print(f"Loading model from checkpoint {checkpoint_path}")
        pl_model_module = PLModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            extractor_args=args.extractor,
            fuser_args=args.fuser,
            identifier_args=args.identifier,
            trainer_args=args.trainer
        )
    else:
        pl_model_module = PLModel(
            extractor_args=args.extractor,
            fuser_args=args.fuser,
            identifier_args=args.identifier,
            trainer_args=args.trainer
        )    
        
    if state_dict_path is not None:
        if is_verbose():
            print(f"Loading model from state_dict {state_dict_path}")
        pl_model_module.load_state_dict(torch.load(state_dict_path))
        
    return pl_model_module
    

def setup_data_and_model(args: Args, state_dict_path: str = None, checkpoint_path: str = None):
    pl_data_module = setup_data_module(args)
    pl_model_module = setup_model_module(args, state_dict_path, checkpoint_path)
    return pl_data_module, pl_model_module


def parser_metrics(metrics_dict: dict):
    """
    Input:
        metrics_dict: dict("metric_name/Te/ad_name": metric_value))
    Output:
        metrics_dict: dict(metric_name: dict(ad_name: metric_value))
    """
    metrics_dict_new = {}
    for k, v in metrics_dict.items():
        k, _, ad_name = k.split('/')
        metrics_dict_new[k] = metrics_dict_new.get(k, {})
        metrics_dict_new[k][ad_name] = v
    return metrics_dict_new

def strip_results_from_cuda(data):
    """
    Recursively move all tensors in the nested structure off cuda,
    and detach and convert to cpu when appropriate.
    Handles:
      - nested dict/list/tuple containing tensors
      - list of strings (unchanged)
    """
    if isinstance(data, dict):
        return {k: strip_results_from_cuda(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # If this is a list of strings, leave it unchanged
        if data and all(isinstance(x, str) for x in data):
            return data
        # Otherwise, recurse
        res = [strip_results_from_cuda(x) for x in data]
        return type(data)(res)  # preserve tuple vs list
    elif torch.is_tensor(data):
        return data.detach().cpu()
    else:
        return data
