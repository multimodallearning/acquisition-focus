import os
import re
from pathlib import Path
from copy import deepcopy
import traceback
import warnings

from typing import List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
from torch._dynamo import OptimizedModule

from scipy.ndimage import gaussian_filter
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image

from acquisition_focus.utils.nifti_utils import nifti_grid_sample
import acquisition_focus.models.segmentation as segmentation_models
from acquisition_focus.utils.python_utils import suppress_stdout

with suppress_stdout():
    import nnunetv2
    from nnunetv2.training.loss import compound_losses
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
    from nnunetv2.utilities.helpers import empty_cache, dummy_context
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    DC_and_CE_loss = compound_losses.DC_and_CE_loss



def load_network(trained_model_path, fold):
    use_folds = [fold] if isinstance(fold, int) else fold # We only trained one fold
    checkpoint_name = "checkpoint_final.pth"

    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(trained_model_path, use_folds, checkpoint_name)

    patch_size = predictor.plans_manager.get_configuration('3d_fullres').patch_size

    return predictor.network, predictor.list_of_parameters, patch_size, predictor.configuration_manager, predictor.allowed_mirroring_axes, predictor.plans_manager, predictor.dataset_json



def get_model_from_network(network, parameters=None):
    model = deepcopy(network)

    if parameters is not None:
        if not isinstance(model, OptimizedModule):
            model.load_state_dict(parameters[0])
        else:
            model._orig_mod.load_state_dict(parameters[0])

    return model



def run_inference_on_image(b_image: torch.Tensor, b_spacing: torch.Tensor,
    network, parameters, plans_manager, configuration_manager, dataset_json,
    inference_allowed_mirroring_axes):

    assert b_image.ndim == 5, f"Expected 5D tensor, got {b_image.ndim}D"
    assert b_spacing.ndim == 2, f"Expected 2D tensor, got {b_spacing.ndim}D"
    assert b_spacing.shape[-1] == 3, f"Expected last dimension to be 3, got {b_spacing.shape[-1]}"
    assert (b_spacing - b_spacing.mean(0)).sum() == 0, "Spacing must be the same for all images"
    B = b_image.shape[0]
    properties = B * [dict(spacing=b_spacing[0].tolist())]

    is_2d_model = len(configuration_manager.patch_size) == 2
    nnunet_spacing = torch.as_tensor(configuration_manager.spacing).to(b_spacing)

    data = []
    for idx_img, img in enumerate(b_image):

        # Resample the image to fit the model requirements
        input_spacing = b_spacing[idx_img]
        img_shape = img.shape[-3:]
        if is_2d_model:
            # Add a dummy spacing
            target_spacing = torch.cat([input_spacing[0:1], nnunet_spacing], dim=0)
        else:
            target_spacing = nnunet_spacing

        target_fov_mm = input_spacing * torch.as_tensor(img_shape).to(target_spacing)
        target_fov_vox = (torch.as_tensor(img_shape).to(b_spacing) * input_spacing / target_spacing).int()

        volume_affine = torch.diag(
            torch.as_tensor(input_spacing.tolist() + [1.])
            * torch.as_tensor([1,-1,1,1]) # TODO: check why inverting middle axis is needed
            )[None]
        resampled, *_ = nifti_grid_sample(img.view(1,1,*img_shape), volume_affine,
            target_fov_mm=target_fov_mm, target_fov_vox=target_fov_vox)

        resampled = (resampled - resampled.mean()) / resampled.std()
        data_elem = dict(
            data=resampled[0],
            data_properites=dict(
                spacing=b_spacing[idx_img].tolist(),
                shape_before_cropping=img_shape,
                bbox_used_for_cropping=torch.stack([torch.zeros(3),torch.as_tensor(img_shape)], dim=0).T.int().tolist(),
                shape_after_cropping_and_before_resampling=img_shape,
            ),
            ofile=None
        )
        data.append(data_elem)

    with suppress_stdout():
        seg = run_inference(data, network, parameters,
            plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes,
            device=b_image.device)

    return torch.as_tensor(np.array(seg)).to(b_image.device)



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



def predict_from_data_iterator(data_iterator,
                               network: nn.Module,
                               parameter_list: List[dict],
                               plans_manager: PlansManager,
                               configuration_manager: ConfigurationManager,
                               dataset_json: dict,
                               inference_allowed_mirroring_axes: Tuple[int, ...],
                               tile_step_size: float = 0.5,
                               use_gaussian: bool = True,
                               use_mirroring: bool = True,
                               perform_everything_on_gpu: bool = True,
                               verbose: bool = True,
                               save_probabilities: bool = False,
                               num_processes_segmentation_export: int=0,
                               device: torch.device = torch.device('cuda')):
    """
    each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
    """
    if num_processes_segmentation_export == 0:
            network = network.to(device)

            r = []
            with torch.no_grad():
                for preprocessed in data_iterator:
                    data = preprocessed['data']
                    if isinstance(data, str):
                        delfile = data
                        data = torch.from_numpy(np.load(data))
                        os.remove(delfile)

                    ofile = preprocessed['ofile']
                    if ofile is not None:
                        print(f'\nPredicting {os.path.basename(ofile)}:')
                    else:
                        print(f'\nPredicting image of shape {data.shape}:')
                    print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

                    properties = preprocessed['data_properites']
                    prediction = predict_logits_from_preprocessed_data(data, network, parameter_list, plans_manager,
                                                                    configuration_manager, dataset_json,
                                                                    inference_allowed_mirroring_axes, tile_step_size,
                                                                    use_gaussian, use_mirroring,
                                                                    perform_everything_on_gpu,
                                                                    verbose, device)
                    prediction = prediction.numpy()

                    if ofile is not None:
                        raise NotImplementedError()
                        # # this needs to go into background processes
                        # # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                        # #                               dataset_json, ofile, save_probabilities)
                        # print('sending off prediction to background worker for resampling and export')
                        # r.append(export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                        #     dataset_json, ofile, save_probabilities))

                    else:
                        label_manager = plans_manager.get_label_manager(dataset_json)

                        r.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
                                    configuration_manager, label_manager,
                                    properties,
                                    save_probabilities))

                    if ofile is not None:
                        print(f'done with {os.path.basename(ofile)}')
                    else:
                        print(f'\nDone with image of shape {data.shape}:')
            ret = r

    else:
        raise NotImplementedError("Multithreaded prediction is disabled.")
    # clear device cache
    empty_cache(device)
    return ret



def predict_logits_from_preprocessed_data(data: torch.Tensor,
                                          network: nn.Module,
                                          parameter_list: List[dict],
                                          plans_manager: PlansManager,
                                          configuration_manager: ConfigurationManager,
                                          dataset_json: dict,
                                          inference_allowed_mirroring_axes: Tuple[int, ...],
                                          tile_step_size: float = 0.5,
                                          use_gaussian: bool = True,
                                          use_mirroring: bool = True,
                                          perform_everything_on_gpu: bool = True,
                                          verbose: bool = True,
                                          device: torch.device = torch.device('cuda')
                                          ) -> torch.Tensor:
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    prediction = None
    overwrite_perform_everything_on_gpu = perform_everything_on_gpu
    if perform_everything_on_gpu:
        try:
            for params in parameter_list:
                # messing with state dict names...
                if not isinstance(network, OptimizedModule):
                    network.load_state_dict(params)
                else:
                    network._orig_mod.load_state_dict(params)

                if prediction is None:
                    prediction = predict_sliding_window_return_logits(
                        network, data, num_seg_heads,
                        configuration_manager.patch_size,
                        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                        tile_step_size=tile_step_size,
                        use_gaussian=use_gaussian,
                        precomputed_gaussian=None,
                        perform_everything_on_gpu=perform_everything_on_gpu,
                        verbose=verbose,
                        device=device)
                else:
                    prediction += predict_sliding_window_return_logits(
                        network, data, num_seg_heads,
                        configuration_manager.patch_size,
                        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                        tile_step_size=tile_step_size,
                        use_gaussian=use_gaussian,
                        precomputed_gaussian=None,
                        perform_everything_on_gpu=perform_everything_on_gpu,
                        verbose=verbose,
                        device=device)
            if len(parameter_list) > 1:
                prediction /= len(parameter_list)

        except RuntimeError:
            print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                  'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
            print('Error:')
            traceback.print_exc()
            prediction = None
            overwrite_perform_everything_on_gpu = False

    if prediction is None:
        for params in parameter_list:
            # messing with state dict names...
            if not isinstance(network, OptimizedModule):
                network.load_state_dict(params)
            else:
                network._orig_mod.load_state_dict(params)

            if prediction is None:
                prediction = predict_sliding_window_return_logits(
                    network, data, num_seg_heads,
                    configuration_manager.patch_size,
                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=None,
                    perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                    verbose=verbose,
                    device=device)
            else:
                prediction += predict_sliding_window_return_logits(
                    network, data, num_seg_heads,
                    configuration_manager.patch_size,
                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=None,
                    perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                    verbose=verbose,
                    device=device)
        if len(parameter_list) > 1:
            prediction /= len(parameter_list)

    print('Prediction done, transferring to CPU if needed')
    prediction = prediction.to('cpu')
    return prediction



def get_sliding_window_generator(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float,
                                 verbose: bool = False):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, 'if tile_size has less entries than image_size, len(tile_size) ' \
                                                      'must be one shorter than len(image_size) (only dimension ' \
                                                      'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)]])
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        if verbose: print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, '
                          f'tile_step_size {tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]])
                    yield slicer



def predict_sliding_window_return_logits(network: nn.Module,
                                         input_image: Union[np.ndarray, torch.Tensor],
                                         num_segmentation_heads: int,
                                         tile_size: Union[Tuple[int, ...], List[int]],
                                         mirror_axes: Tuple[int, ...] = None,
                                         tile_step_size: float = 0.5,
                                         use_gaussian: bool = True,
                                         precomputed_gaussian: torch.Tensor = None,
                                         perform_everything_on_gpu: bool = True,
                                         verbose: bool = True,
                                         device: torch.device = torch.device('cuda')) -> Union[np.ndarray, torch.Tensor]:
    if perform_everything_on_gpu:
        assert device.type == 'cuda', 'Can use perform_everything_on_gpu=True only when device="cuda"'

    network = network.to(device)
    network.eval()

    empty_cache(device)

    with torch.no_grad():
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print('WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...')
                perform_everything_on_gpu = False

            results_device = device if perform_everything_on_gpu else torch.device('cpu')

            if verbose: print("step_size:", tile_step_size)
            if verbose: print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, tile_size, 'constant', {'value': 0}, True, None)

            if use_gaussian:
                gaussian = compute_gaussian(tuple(tile_size), sigma_scale=1. / 8, value_scaling_factor=1000,
                                            device=device) if precomputed_gaussian is None else precomputed_gaussian

            slicers = get_sliding_window_generator(data.shape[1:], tile_size, tile_step_size, verbose=verbose)

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = torch.device('cpu')
                predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]), dtype=torch.half,
                                               device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                            device=results_device)
                gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)

            for sl in slicers:
                workon = data[sl][None]
                workon = workon.to(device, non_blocking=False)

                prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(results_device)

                predicted_logits[sl] += (prediction * gaussian if use_gaussian else prediction)
                n_predictions[sl[1:]] += (gaussian if use_gaussian else 1)

            predicted_logits /= n_predictions
    empty_cache(device)
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None) \
        -> torch.Tensor:
    prediction = network(x)

    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction

def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).type(dtype).to(device)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map



def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps



def inject_dg_trainers_into_nnunet(num_epochs=1000):
    dg_trainer_paths = Path(segmentation_models.__file__).parent.glob("*.py")
    target_dir = Path(nnunetv2.__path__[0], "training/nnUNetTrainer/variants/acquisition_focus/")
    target_dir.mkdir(exist_ok=True)

    for tr in dg_trainer_paths:
        # open file
        with open(tr, "r") as f:
            tr_code = f.read()
        tr_code_with_set_epochs = re.sub(
            r"self\.num_epochs = \d+", f"self.num_epochs = {num_epochs}", tr_code
        )

        with open(target_dir / tr.name, "w") as f:
            f.write(tr_code_with_set_epochs)



def get_segment_fn(trained_model_path, fold=0):

    # trained_model_path: Like 'nnUNetV2_results/Dataset_xxx/nnUNetTrainer__nnUNetPlans__2d'

    inject_dg_trainers_into_nnunet()
    (network, parameters, patch_size, configuration_manager,
    inference_allowed_mirroring_axes, plans_manager, dataset_json) = load_network(trained_model_path, fold)

    def segment_closure(b_image: torch.Tensor, b_spacing: torch.Tensor):
        assert b_image.ndim == 5, f"Expected 5D tensor, got {b_image.ndim}"
        return run_inference_on_image(b_image, b_spacing, network, parameters,
            plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes
        )

    return segment_closure