from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression

def pearson_corr(preds, target):
    return pearsonr(preds, target)[0]

def spearman_corr(preds, target):
    return spearmanr(preds, target)[0]

def mutual_info_corr(preds, target):
    return mutual_info_regression(preds.reshape(-1, 1), target)[0]

def kendall_corr(preds, target):
    return kendalltau(preds, target)[0]

def regression_metrics(preds, target):
    return {
        'pearson_corr': pearson_corr(preds, target),
        'spearman_corr': spearman_corr(preds, target),
        'kendall_corr': kendall_corr(preds, target),
        'mutual_info_corr': mutual_info_corr(preds, target)
    }
