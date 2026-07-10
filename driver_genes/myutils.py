import os, yaml
import warnings

# Global verbosity flag for controlling package-wide print behavior
VERBOSE: bool = False


def set_verbosity(verbose: bool = True):
    """Set global verbosity for the driver_genes package."""
    global VERBOSE
    VERBOSE = bool(verbose)


def is_verbose() -> bool:
    """Return current global verbosity setting."""
    return VERBOSE


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def getall(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}


def dict_to_args(d):
    if isinstance(d, Args):
        return d
    if d is None:
        return None
    
    x = Args()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_to_args(v))
        else:
            setattr(x, k, v)
    return x


def args_to_dict(x):
    if isinstance(x, dict):
        return x
    if x is None:
        return None
    
    d = {}
    for k, v in x.__dict__.items():
        if isinstance(v, Args):
            d[k] = args_to_dict(v)
        else:
            d[k] = v
    return d


def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def paste_paths(root_dir: str, file_name: str | list[str]):
    if isinstance(file_name, str):
        return os.path.join(root_dir, file_name)
    elif isinstance(file_name, list):
        return [paste_paths(root_dir, f) for f in file_name]
    else:
        raise ValueError(f"Invalid file_name: {file_name}")


def reduce_dict_list(d_list: list[dict], method: str = 'mean'):
    d_total = {}
    n_total = {}
    for d in d_list:
        for k, v in d.items():
            d_total[k] = d_total.get(k, 0) + v
            n_total[k] = n_total.get(k, 0) + 1
    if method == 'mean':
        return {k: d_total[k] / n_total[k] for k in d_total.keys()}
    elif method == 'sum':
        return {k: d_total[k] for k in d_total.keys()}
    else:
        raise ValueError(f"Invalid method: {method}")
    
    
    
def render_template(template_path: str, **kwargs):
    with open(template_path, 'r') as f:
        template = f.read()
    return yaml.safe_load(template.format(**kwargs))


def parser_metrics(metrics_dict: dict):
    """
    Input:
        metrics_dict: dict("metric_name/Te/ad_name": metric_value))
    Output:
        metrics_dict: dict(metric_name: dict(ad_name: metric_value))
    """
    warnings.warn("parser_metrics is deprecated. Use trainer.parser_metrics instead.")
    metrics_dict_new = {}
    for k, v in metrics_dict.items():
        k, _, ad_name = k.split('/')
        metrics_dict_new[k] = metrics_dict_new.get(k, {})
        metrics_dict_new[k][ad_name] = v
    return metrics_dict_new