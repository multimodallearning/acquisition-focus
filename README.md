# GitHub repository for 'AcquisitionFocus: Joint Optimization of Acquisition Orientation and Cardiac Volume Reconstruction Using Deep Learning'

# Installation
```bash
git clone git@github.com:multimodallearning/acquisition-focus.git
cd acquisition-focus
poetry install
```

To preprocess the data, c3d is needed:
http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D

# Preprocessing
## Preprocessing MMWHS
Put base files in this folder:
```acquisition-focus/data/datasets/MMWHS/base_files```

Example:
```
acquisition-focus/data/datasets/MMWHS/base_files/mr_train/mr_train_1001_image.nii.gz
acquisition-focus/data/datasets/MMWHS/base_files/mr_train/mr_train_1001_label.nii.gz
...
acquisition-focus/data/datasets/MMWHS/base_files/mr_train/mr_train_1020_image.nii.gz
acquisition-focus/data/datasets/MMWHS/base_files/mr_train/mr_train_1020_label.nii.gz
```

Open the preprocessing notebook to prepare files:
```/home/weihsbach/storage/staff/christianweihsbach/acquisition_focus/acquisition_focus/preprocessing/preprocess_mmwhs.ipynb```

## Preprocessing MRXCAT
Put base files in this folder:
```acquisition-focus/data/datasets/MRXCAT/base_files```

Example:
```
acquisition-focus/data/datasets/MRXCAT/base_files/phantom_001/phantom_001_novessels_labels/phantom_act.nii.gz
acquisition-focus/data/datasets/MRXCAT/base_files/phantom_001/phantom_001_novessels_texture/phantom.nii.gz
acquisition-focus/data/datasets/MRXCAT/base_files/phantom_004/phantom_004_novessels_labels/phantom_act.nii.gz
acquisition-focus/data/datasets/MRXCAT/base_files/phantom_004/phantom_004_novessels_texture/phantom.nii.gz
```

Open the preprocessing notebook to prepare files:
```/home/weihsbach/storage/staff/christianweihsbach/acquisition_focus/acquisition_focus/preprocessing/preprocess_mrxcat.ipynb```

# Run view optimization
The script will run a three-fold cross-validation with several optimization stages, each optimizing one of the requested input views. Outputs can be found in the `acquisition-focus/data/output` directory. All metrics will be saved to a wandb (https://wandb.ai/home) dashboard.

```bash
python main_acquisition_focus.py
```

# Configure view optimization
The script can be configured using the `config_dict.json` inside the project's main directory:

```json
{
    "num_folds": 3, # The number of folds that is used during cross-validation
    "fold_override": 0, # The specific fold that should be trained. If null, all folds are trained
    "epochs": 80,
    "test_only_and_output_to": null, # Directory to which the test data output is stored to. You also should pass `model_checkpoint_path` and `transform_model_checkpoint_path` in that case.
    "batch_size": 2,
    "num_grad_accum_steps": 2, # Number of batches to accumulate gradients for (results in batch_size 4 for this specific case)
    "val_batch_size": 1,
    "do_augment_input_orientation": true, # Augment the base view orientation (this augments the cutting orientation)
    "do_augment_recon_orientation": false, # Augment the orientation inside the reconstruction model (this disaligns the view content with regard to the 3D embedding)
    "aug_phases": ["train", "val"],
    "sample_augment_strength": 1.0, # Relative augment strength to be used
    "use_affine_theta": null, # Will be set inside the specific optimization stage.
    "base_views": ["p2CH", "p2CH", "p2CH"], # The initial views that are optimized. Change the initial view extracted from /acquisition-focus/functional/clinical_cardiac_views.py:get_clinical_cardiac_view_affines(). The more base views in this list, the more views are optimized during training
    "offset_clip_value": 0.2, # The +- view offset that is allowed during optimization
    "zoom_clip_value": 0.0, # The +- zoom percentage that is allowed during optimization
    "affine_theta_optim_method": "R6-vector",
    "view_optimization_mode": "opt-all", # This is set during the individual optimization stages
    "use_binarized_labels": false, # Whether to perform binary reconstruction
    "dataset": [
        "mmwhs", # dataset id
        "./data/datasets/MMWHS/prepared_files" # dataset files
    ],
    "hires_fov_mm": [ # The reconstruction FOV
        192.0,
        192.0,
        192.0
    ],
    "hires_fov_vox": [ # The reconstruction vox count (these settings result in 1.5mm spacing)
        128,
        128,
        128
    ],
    "slice_fov_mm": [
        192.0,
        192.0,
        1.5
    ],
    "slice_fov_vox": [
        128,
        128,
        1
    ],
    "prescan_fov_mm": [ # The prescan FOV (that is used to determine the optimal slice orientation)
        192.0,
        192.0,
        192.0
    ],
    "prescan_fov_vox": [
        128,
        128,
        128
    ],
    "clinical_view_affine_type": "from-gt", # Whether to extract clinical view affines "from-gt" 3D labels or "from-segmented" 3d labels
    "label_slice_type": "from-gt", # Whether to reconstruct "from-gt" 2D labels or "from-segmented" 2D labels
    "optimize_lv_only": false, # Flag to only optimize the LV class reconstruction
    "rotate_slice_to_min_principle": false, # Flag to rotate the 2D slice in-plane along the main mass axis
    "lr": 0.001,
    "use_scheduling": true,
    "model_type": "hybrid-unet",
    "save_every": "best", # Save models (at "best" reconstruction outcome) or every N epochs (e.g. N=20)
    "mdl_save_prefix": "data/models", # Sub path to store modes to
    "debug": false,
    "wandb_mode": "online", # Controls wandb logging ("online", "disabled")
    "sweep_type": "stage-sweep", # Sweep mode (single-run: "null", multi-stage-optimization: "stage-sweep")
    "stage_override": null, # Select a specific stage of the multi-stage-optimization by its name
    "model_checkpoint_path": null, # Override this to load a previously stored reconstruction model
    "transform_model_checkpoint_path": null, # Override this to load a previously stored view optimization model
    "do_output": true, # Save output files
    "device": "cuda",
    "use_autocast": false,
    "use_caching": true # Cache datasets for same config_dict args and the current git commit
}

```
# Citation
Weihsbach, C., Vogt, N., Al-Haj Hemidi, Z., Bigalke, A., Hansen, L., Oster, J., & Heinrich, M. P. (2024). AcquisitionFocus: Joint Optimization of Acquisition Orientation and Cardiac Volume Reconstruction Using Deep Learning. Sensors, 24(7), 2296.