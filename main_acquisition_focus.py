import os
from pathlib import Path
import json
from datetime import datetime
import argparse

import dill
from git import Repo
import joblib
import randomname
import wandb

from pytorch_run_on_recommended_gpu.run_on_recommended_gpu import get_cuda_environ_vars as get_vars
os.environ.update(get_vars(os.environ.get('CUDA_VISIBLE_DEVICES','0')))
import torch

from acquisition_focus.running.stages import get_std_stages
from acquisition_focus.datasets.mmwhs_dataset import MMWHSDataset
from acquisition_focus.datasets.mrxcat_dataset import MRXCATDataset
from acquisition_focus.utils.log_utils import get_fold_postfix
from acquisition_focus.utils.python_utils import DotDict, get_script_dir
from acquisition_focus.running.run_dl import run_dl



os.environ['CACHE_PATH'] = str(Path(get_script_dir(), '.cache'))

def prepare_data(config):
    args = [config.dataset[1]]

    if config.dataset[0] == 'mmwhs':
        dataset_class = MMWHSDataset
    elif config.dataset[0] == 'mrxcat':
        dataset_class = MRXCATDataset
    else:
        raise ValueError()

    kwargs = {k:v for k,v in config.items()}

    cache_dir = 'git-' + config.git_commit.replace('!', '')
    cache_path = Path(os.environ['CACHE_PATH'], cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    arghash = joblib.hash(joblib.hash(args)+joblib.hash(kwargs))
    hashfile = cache_path / f"argshash_{arghash}_dataset.dil"

    if config.use_caching:
        if hashfile.is_file():
            print("Loading dataset from cache:", hashfile)
            with open(hashfile, 'rb') as file:
                dataset = dill.load(file)
        else:
            dataset = dataset_class(*args, **kwargs)
            print("Caching dataset:", hashfile)
            with open(hashfile, 'wb') as file:
                dill.dump(dataset, file)
    else:
        dataset = dataset_class(*args, **kwargs)

    return dataset



def normal_run(run_name, config_dict, fold_properties, training_dataset, test_dataset, run_test_once_only):
    with wandb.init(project=PROJECT_NAME, group="training", job_type="train",
            config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        ) as run:
        run.name = run_name
        print("Running", run.name)
        config = wandb.config
        run_dl(get_script_dir(), config, fold_properties,
               training_dataset=training_dataset, test_dataset=test_dataset,
               run_test_once_only=run_test_once_only)



def stage_sweep_run(run_name, config_dict, fold_properties, all_stages, training_dataset, test_dataset, run_test_once_only):
    for stage in all_stages:
        stg_id = all_stages.current_key

        # Prepare stage settings
        stage.activate()

        stage_config = config_dict.copy()
        # Update intersecting keys of both
        stage_config.update((key, stage[key]) for key in set(stage).intersection(stage_config))
        print()

        torch.cuda.empty_cache()
        with wandb.init(project=PROJECT_NAME, config=stage_config, settings=wandb.Settings(start_method="thread"),
            mode=stage_config['wandb_mode']) as run:

            run.name = f"{run_name}_stage-{stg_id}"
            print("Running", run.name)
            config = wandb.config

            run_dl(get_script_dir(), config, fold_properties, stage, training_dataset, test_dataset, run_test_once_only)
        wandb.finish()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    PROJECT_NAME = "acquisition_focus"

    # Add argument parser for additional config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_config_path', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    if args.meta_config_path is not None:
        with open(args.meta_config_path, 'r') as f:
            meta_config_dict = DotDict(json.load(f))
    else:
        meta_config_dict = DotDict()

    now_string = datetime.now().strftime("%Y%m%d__%H_%M_%S")
    this_repo = Repo(get_script_dir())

    with open(Path(get_script_dir(), 'config_dict.json'), 'r') as f:
        config_dict = DotDict(json.load(f))

    # Merge meta config
    config_dict.update(meta_config_dict)

    # Log commmit id and dirtiness
    dirty_str = "!dirty-" if this_repo.is_dirty() else ""
    config_dict['git_commit'] = f"{dirty_str}{this_repo.commit().hexsha}"

    run_test_once_only = not (config_dict.test_only_and_output_to in ["", None])

    train_config = DotDict(config_dict.copy())

    if run_test_once_only:
        train_config['state'] = 'empty'
    training_dataset = prepare_data(train_config)

    test_config = DotDict(config_dict.copy())
    test_config['state'] = 'test'
    test_dataset = prepare_data(test_config)

    # Configure folds
    if config_dict.num_folds < 1:
        train_idxs = range(len(training_dataset))
        val_idxs = []
        fold_idx = -1
        fold_iter = ([fold_idx, (train_idxs, val_idxs)],)

    else:
        fold_iter = []
        for fold_idx in range(config_dict.num_folds):
            current_fold_idxs = training_dataset.data_split['train_folds'][f"fold_{fold_idx}"]
            train_files = [training_dataset.data_split['train_files'][idx] for idx in current_fold_idxs['train_idxs']]
            val_files = [training_dataset.data_split['train_files'][idx] for idx in current_fold_idxs['val_idxs']]

            train_ids = set([training_dataset.get_file_id(fl)[0] for fl in train_files])
            val_ids = set([training_dataset.get_file_id(fl)[0] for fl in val_files])
            assert len(train_ids.intersection(val_ids)) == 0, \
                f"Training and validation set must not overlap. But they do: {train_ids.intersection(val_ids)}"
            train_idxs = training_dataset.switch_3d_identifiers(train_ids)
            val_idxs = training_dataset.switch_3d_identifiers(val_ids)
            fold_iter.append((
                [idx for idx in train_idxs if idx is not None],
                [idx for idx in val_idxs if idx is not None]
            ))
        fold_iter = list(enumerate(fold_iter))

        if config_dict['fold_override'] is not None:
            selected_fold = config_dict['fold_override']
            fold_iter = fold_iter[selected_fold:selected_fold+1]

    rnd_name = randomname.get_name()
    run_name = f"{now_string}_{rnd_name}"

    for fold_properties in fold_iter:
        run_name_with_fold = run_name + f"_{get_fold_postfix(fold_properties)}"
        if config_dict['sweep_type'] is None:
            normal_run(run_name_with_fold, config_dict, fold_properties, training_dataset, test_dataset)

        elif config_dict['sweep_type'] == 'stage-sweep':
            stage_iterator = get_std_stages(config_dict)

            stage_sweep_run(run_name_with_fold, config_dict, fold_properties, stage_iterator,
                            training_dataset=training_dataset, test_dataset=test_dataset, run_test_once_only=run_test_once_only)

        else:
            raise ValueError()

        if config_dict.debug or run_test_once_only:
            break
        # End of fold loop