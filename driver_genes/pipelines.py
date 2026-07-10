from scanpy import AnnData
import os, warnings, sys, tqdm, glob
import torch
import numpy as np
warnings.filterwarnings('ignore')

from .trainer import (
    prepare_config,
    setup_trainer,
    setup_data_and_model,
    setup_data_module,
    setup_model_module,
    build_predict_loader,
    parser_metrics
)
from .data.utils import reassemble_anndata, concat_results
from .myutils import is_verbose
import pandas as pd


class Pipeline:
    def __init__(self, 
                 config_path: str | None = None, 
                 config_dict: dict | None = None, 
                 fitted: bool = False,
                 **kwargs):
        self.args = prepare_config(
            config_path=config_path, 
            config_dict=config_dict, 
            rerun=fitted, 
            **kwargs
        )
        self.fitted = fitted
        self.initialize()
        if fitted:
            self.load_model(from_best=True)
        
        
    def initialize(self):
        self.data_module, self.model_module = setup_data_and_model(self.args)
        self.trainer = setup_trainer(**self.args.trainer.getall())
        
        
    def reload_data(self):
        self.data_module = setup_data_module(self.args)
        
    def reset_model(self):
        self.model_module = setup_model_module(self.args)
        
    def load_model(self, 
                   state_dict_path: str = None, 
                   checkpoint_path: str = None, 
                   from_best: bool = False):
        if from_best:
            if self.trainer.checkpoint_callback.best_model_path:
                checkpoint_path = self.trainer.checkpoint_callback.best_model_path
            elif os.path.exists(os.path.join(self.args.trainer.output_dir, 'best_model.pth')):
                state_dict_path = os.path.join(self.args.trainer.output_dir, 'best_model.pth')
            else:
                raise ValueError("No best model found")
        if checkpoint_path is not None:
            if is_verbose():
                print(f"Loading model from checkpoint {checkpoint_path}")
            self.model_module = setup_model_module(self.args, checkpoint_path=checkpoint_path)
        elif state_dict_path is not None:
            if is_verbose():
                print(f"Loading model from state_dict {state_dict_path}")
            self.model_module = setup_model_module(self.args, state_dict_path=state_dict_path)
        else:
            raise ValueError("Either checkpoint_path or state_dict_path must be provided")
            
            
    def fit(self):
        self.trainer.fit(self.model_module, self.data_module)
        self.fitted = True
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path
        self.model_module = setup_model_module(self.args, checkpoint_path=self.best_model_path)
        
        
    def save_best_model(self, 
                        path: str = None, 
                        remove_checkpoints: bool = True) -> str:
        if path is None:
            path = os.path.join(self.args.trainer.output_dir, 'best_model.pth')
        if is_verbose():
            print(f"Saving best model to {path}")
        torch.save(self.model_module.state_dict(), path)
        if remove_checkpoints:
            for f in glob.glob(os.path.join(self.args.trainer.output_dir, 'checkpoints', '*.ckpt')):
                os.remove(f)
            if is_verbose():
                print(f"Removing checkpoints from {self.args.trainer.output_dir}/checkpoints/ to reduce disk usage")
        return path
    
    
    def evaluate(self, 
                 num_results: int = 1, 
                 seed: int | None = 0, 
                 concat: bool = False, 
                 agg: bool = False) -> tuple[pd.DataFrame, dict[str, AnnData]]:
        if seed is None:
            seed = np.random.randint(0, 1000000)
        epoch_metric_list = []
        test_prediction_list = []
        for s in range(seed, seed + num_results):
            self.data_module.initialize(dataset_name='test', seed=s)
            test_metrics = self.trainer.test(self.model_module, self.data_module.test_dataloader(), verbose=is_verbose())[0]
            test_predictions = self.model_module.predictions
            epoch_metric_list.append(pd.DataFrame(parser_metrics(test_metrics)).assign(seed=s))
            test_prediction_list.append(test_predictions)
        epoch_metric_df = pd.concat(epoch_metric_list)
        epoch_metric_df['job_name'] = self.args.trainer.job_name
        epoch_metric_df['version'] = self.args.trainer.version
        
        # concat the predictions of multiple runs
        if concat:
            test_prediction_list = [concat_results(test_prediction_list)]
            
        # reassemble the predictions into adatas
        test_pred_adatas_list = [
            reassemble_anndata(
                ds_dict=self.data_module.ds_dict['test'],
                pred_dict=test_prediction,
                agg=agg # whether to aggregate the predictions for the same cell in multiple runs
            ) for test_prediction in test_prediction_list
        ]
        if len(test_pred_adatas_list) == 1:
            test_pred_adatas_list = test_pred_adatas_list[0]
            
        # annotate which datasets are in-domain and out-of-domain
        train_datasets = self.data_module.ds_dict['train'].keys()
        epoch_metric_df['Group'] = epoch_metric_df.index.map(
            lambda idx: 
                'All' if idx=='a' else 
                'ID' if idx in train_datasets else 
                'OOD'
        )
        epoch_metric_df.reset_index(names='prefix', inplace=True)

        return epoch_metric_df, test_pred_adatas_list
    
    
    def predict(self, 
                num_results: int = 1, 
                seed: int | None = 0) -> dict[str, AnnData]:
        if seed is None:
            seed = np.random.randint(0, 1000000)
        if 'pred' in self.data_module.ds_dict.keys():
            split = 'pred'
        else:
            warnings.warn("No prediction split found in the data module, using the test split instead")
            split = 'test'
        ds_dict = self.data_module.ds_dict[split]
        prediction_list = []
        for s in range(seed, seed + num_results):
            self.data_module.initialize(dataset_name=split, seed=s)
            pred_dataloader = build_predict_loader(
                ds_dict, 
                self.args.dataset.batch_size, 
                self.args.dataset.num_workers
            )
            self.trainer.predict(self.model_module, pred_dataloader)
            # the prediction func in pl can only handle sequential dataloader
            # and loses the ad_name information, 
            # so we need leverage the ds_dict to convert the predictions to a named dict
            prediction_list.append(dict(zip(
                ds_dict.keys(), 
                self.model_module.predictions.values()
            )))
        predictions = concat_results(prediction_list)
        pred_adatas = reassemble_anndata(
            ds_dict=ds_dict,
            pred_dict=predictions,
            agg=True
        )
        return pred_adatas
    