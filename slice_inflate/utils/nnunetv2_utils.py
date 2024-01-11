import os
import sys
from copy import deepcopy
from contextlib import contextmanager

from torch._dynamo import OptimizedModule
import torch

from nnunetv2.inference.predict_from_raw_data import load_trained_model_for_inference, predict_from_data_iterator


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_network(model_training_output_path, fold):
    use_folds = [fold] if isinstance(fold, int) else fold # We only trained one fold
    checkpoint_name = "checkpoint_final.pth"

    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_trained_model_for_inference(model_training_output_path, use_folds, checkpoint_name)

    patch_size = plans_manager.get_configuration('3d_fullres').patch_size

    return network, parameters, patch_size, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json

def get_model_from_network(network, parameters=None):
    model = deepcopy(network)

    if parameters is not None:
        if not isinstance(model, OptimizedModule):
            model.load_state_dict(parameters[0])
        else:
            model._orig_mod.load_state_dict(parameters[0])

    return model

from nnunetv2.inference.data_iterators import get_data_iterator_from_raw_npy_data

def run_inference_on_image(b_image: torch.Tensor, b_spacing: torch.Tensor,
    network, parameters, plans_manager, configuration_manager, dataset_json,
    inference_allowed_mirroring_axes):

    # assert b_image.ndim == 5, f"Expected 5D tensor, got {b_image.ndim}D"
    # assert b_spacing.ndim == 2, f"Expected 2D tensor, got {b_spacing.ndim}D"
    # assert b_spacing.shape[-1] == 3, f"Expected last dimension to be 3, got {b_spacing.shape[-1]}"
    assert (b_spacing - b_spacing.mean(0)).sum() == 0, "Spacing must be the same for all images"

    properties = dict(spacing=b_spacing[0].tolist())

    with suppress_stdout():
        data = get_data_iterator_from_raw_npy_data(b_image.squeeze(1).cpu().numpy(), None, [properties], None,
                                                   plans_manager, dataset_json, configuration_manager,
                                                   num_processes=1, pin_memory=False)
        seg = run_inference(data, network, parameters,
            plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes,
            device=b_image.device)
    return torch.as_tensor(seg[0]).to(b_image.device)


def run_inference(data, network, parameters,
                      plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes,
                      device):

    tile_step_size = 0.5
    use_gaussian = True
    use_mirroring = True
    save_probabilities = False
    verbose = False
    perform_everything_on_gpu = device.type == 'cuda'
    num_processes_segmentation_export = 0

    network = deepcopy(network)
    return predict_from_data_iterator(data, network, parameters, plans_manager, configuration_manager, dataset_json,
                                inference_allowed_mirroring_axes, tile_step_size, use_gaussian, use_mirroring,
                                perform_everything_on_gpu, verbose, save_probabilities,
                                num_processes_segmentation_export, device)


def get_segment_fn(model_training_output_path, fold, device):

    (network, parameters, patch_size, configuration_manager,
     inference_allowed_mirroring_axes, plans_manager, dataset_json) = load_network(model_training_output_path, fold)

    def segment_closure(b_image: torch.Tensor, b_spacing: torch.Tensor):
        # assert b_image.ndim == 5, f"Expected 5D tensor, got {b_image.ndim}"
        return run_inference_on_image(b_image, b_spacing, network, parameters,
            plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes
        )

    return segment_closure