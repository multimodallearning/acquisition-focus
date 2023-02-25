import wandb
import numpy as np
import torch

def get_global_idx(fold_idx, epoch_idx, max_epochs):
    # Get global index e.g. 2250 for fold_idx=2, epoch_idx=250 @ max_epochs<1000
    fold_idx = max(0, fold_idx)
    return 10**len(str(int(max_epochs)))*fold_idx + epoch_idx



def log_label_metrics(log_prefix, log_postfix, metrics, log_idx,
    logger_selected_metrics=('dice'), print_selected_metrics=('dice')):
    for m_name, m_content in metrics.items():
        for tag in m_content.keys():
            log_path = f"{log_prefix}_{m_name}_{tag}{log_postfix}"

            if m_name in logger_selected_metrics:
                wandb.log({log_path: m_content[tag]}, step=log_idx)
            if m_name in print_selected_metrics:
                print(log_path, f"{m_content[tag]:.3f}")



def log_oa_metrics(log_prefix, log_postfix, metrics, log_idx,
    logger_selected_metrics=('dice'), print_selected_metrics=('dice')):
    for m_name, m_content in metrics.items():
        log_path = f"{log_prefix}_{m_name}{log_postfix}"

        if m_name in logger_selected_metrics:
            wandb.log({log_path: m_content}, step=log_idx)
        if m_name in print_selected_metrics:
            print(log_path, f"{m_content:.3f}")



def log_affine_param_stats(log_prefix, log_postfix, affine_params_dct, log_idx,
    logger_selected_metrics=('mean'), print_selected_metrics=('mean')):

    means = dict()

    for param_name, param in affine_params_dct.items():
        stats = {}
        stackk = torch.cat(param, dim=0)
        stats['mean'] = stackk.mean(0)
        stats['std'] = stackk.std(0)
        means[param_name] = stats['mean']

        for tag in ('mean', 'std'):
            log_path = f"{log_prefix}{param_name}PLX_{tag}{log_postfix}"

            if tag in logger_selected_metrics:
                log_dict = {log_path.replace('PLX', str(val_idx)): val \
                    for val_idx,val in enumerate(stats[tag])}
                wandb.log(log_dict, step=log_idx)

            if tag in print_selected_metrics:
                log_path = log_path.replace('PLX', '')
                print(log_path, ' '.join([f"{p:.3f}" for p in stats[tag].tolist()]))

    return means['theta_ap'], means['theta_tp']
