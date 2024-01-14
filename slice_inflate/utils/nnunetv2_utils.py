import os
import sys
from copy import deepcopy
from contextlib import contextmanager

from torch._dynamo import OptimizedModule
import torch

from nnunetv2.inference.predict_from_raw_data import load_trained_model_for_inference #,predict_from_data_iterator
# from nnunetv2.inference.data_iterators import get_data_iterator_from_raw_npy_data

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



def run_inference_on_image(b_image: torch.Tensor, b_spacing: torch.Tensor,
    network, parameters, plans_manager, configuration_manager, dataset_json,
    inference_allowed_mirroring_axes):

    # assert b_image.ndim == 5, f"Expected 5D tensor, got {b_image.ndim}D"
    # assert b_spacing.ndim == 2, f"Expected 2D tensor, got {b_spacing.ndim}D"
    # assert b_spacing.shape[-1] == 3, f"Expected last dimension to be 3, got {b_spacing.shape[-1]}"
    assert (b_spacing - b_spacing.mean(0)).sum() == 0, "Spacing must be the same for all images"
    B = b_image.shape[0]
    properties = B * [dict(spacing=b_spacing[0].tolist())]

    with suppress_stdout():
        data = get_data_iterator_from_raw_npy_data([im for im in b_image.cpu().numpy()], None, properties, None,
                                                   plans_manager, dataset_json, configuration_manager,
                                                   num_processes=NUM_PROCESSES, pin_memory=False)
        seg = run_inference(data, network, parameters,
            plans_manager, configuration_manager, dataset_json, inference_allowed_mirroring_axes,
            device=b_image.device)
    return torch.as_tensor(seg).to(b_image.device)


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


from typing import List, Tuple, Union
import numpy as np
import  torch.nn as nn
import  torch.nn.functional as F
default_num_processes = 0
import warnings
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import traceback
from scipy.ndimage import gaussian_filter
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.utilities.label_handling.label_handling import LabelManager

def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple([slice(*i) for i in bounding_box])

def get_data_iterator_from_raw_npy_data(image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        plans_manager: PlansManager,
                                        dataset_json: dict,
                                        configuration_manager: ConfigurationManager,
                                        num_processes: int = 3,
                                        pin_memory: bool = False
                                        ):
    list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
        image_or_list_of_images

    if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
        segs_from_prev_stage_or_list_of_segs_from_prev_stage = [segs_from_prev_stage_or_list_of_segs_from_prev_stage]

    if isinstance(truncated_ofname, str):
        truncated_ofname = [truncated_ofname]

    if isinstance(properties_or_list_of_properties, dict):
        properties_or_list_of_properties = [properties_or_list_of_properties]

    num_processes = min(num_processes, len(list_of_images))
    ppa = PreprocessAdapterFromNpy(list_of_images, segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                   properties_or_list_of_properties, truncated_ofname,
                                   plans_manager, dataset_json, configuration_manager, num_processes)
    if num_processes == 0:
        mta = SingleThreadedAugmenter(ppa, None)
    else:
        raise NotImplementedError()
        # mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
    return mta


class SingleThreadedAugmenter(object):
    """
    Use this for debugging custom transforms. It does not use a background thread and you can therefore easily debug
    into your augmentations. This should not be used for training. If you want a generator that uses (a) background
    process(es), use MultiThreadedAugmenter.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
    """
    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.data_loader)
        if self.transform is not None:
            item = self.transform(**item)
        return item

    def next(self):
        return self.__next__()



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
                               num_processes_segmentation_export: int = default_num_processes,
                               device: torch.device = torch.device('cuda')):
    """
    each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
    """
    if num_processes_segmentation_export == 0 or 'NNUNET_DEBUG_FLAG' in os.environ:
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
        raise NotImplementedError()
        # with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        #     network = network.to(device)

        #     r = []
        #     with torch.no_grad():
        #         for preprocessed in data_iterator:
        #             data = preprocessed['data']
        #             if isinstance(data, str):
        #                 delfile = data
        #                 data = torch.from_numpy(np.load(data))
        #                 os.remove(delfile)

        #             ofile = preprocessed['ofile']
        #             if ofile is not None:
        #                 print(f'\nPredicting {os.path.basename(ofile)}:')
        #             else:
        #                 print(f'\nPredicting image of shape {data.shape}:')
        #             print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

        #             properties = preprocessed['data_properites']

        #             # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
        #             # npy files
        #             proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))
        #             while not proceed:
        #                 sleep(0.1)
        #                 proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))

        #             prediction = predict_logits_from_preprocessed_data(data, network, parameter_list, plans_manager,
        #                                                             configuration_manager, dataset_json,
        #                                                             inference_allowed_mirroring_axes, tile_step_size,
        #                                                             use_gaussian, use_mirroring,
        #                                                             perform_everything_on_gpu,
        #                                                             verbose, device)
        #             prediction = prediction.numpy()

        #             if ofile is not None:
        #                 # this needs to go into background processes
        #                 # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
        #                 #                               dataset_json, ofile, save_probabilities)
        #                 print('sending off prediction to background worker for resampling and export')
        #                 r.append(
        #                     export_pool.starmap_async(
        #                         export_prediction_from_logits, ((prediction, properties, configuration_manager, plans_manager,
        #                                                         dataset_json, ofile, save_probabilities),)
        #                     )
        #                 )
        #             else:
        #                 label_manager = plans_manager.get_label_manager(dataset_json)
        #                 # convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
        #                 #                                                             configuration_manager, label_manager,
        #                 #                                                             properties,
        #                 #                                                             save_probabilities)
        #                 r.append(
        #                     export_pool.starmap_async(
        #                         convert_predicted_logits_to_segmentation_with_correct_shape, (
        #                             (prediction, plans_manager,
        #                             configuration_manager, label_manager,
        #                             properties,
        #                             save_probabilities),)
        #                     )
        #                 )
        #             if ofile is not None:
        #                 print(f'done with {os.path.basename(ofile)}')
        #             else:
        #                 print(f'\nDone with image of shape {data.shape}:')
        #     ret = [i.get()[0] for i in r]

        # if isinstance(data_iterator, MultiThreadedAugmenter):
        #     data_iterator._finish()

    # clear lru cache
    compute_gaussian.cache_clear()
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
    """
    IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
    TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!
    """
    # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
    # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
    # things a lot faster for some datasets.
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



def pad_nd_image(image: Union[torch.Tensor, np.ndarray], new_shape: Tuple[int, ...] = None,
                 mode: str = "constant", kwargs: dict = None, return_slicer: bool = False,
                 shape_must_be_divisible_by: Union[int, Tuple[int, ...], List[int]] = None) -> \
        Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Tuple]]:
    """
    One padder to pad them all. Documentation? Well okay. A little bit

    Padding is done such that the original content will be at the center of the padded image. If the amount of padding
    needed it odd, the padding 'above' the content is larger,
    Example:
    old shape: [ 3 34 55  3]
    new_shape: [3, 34, 96, 64]
    amount of padding (low, high for each axis): [[0, 0], [0, 0], [20, 21], [30, 31]]

    :param image: can either be a numpy array or a torch.Tensor. pad_nd_image uses np.pad for the former and
           torch.nn.functional.pad for the latter
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
           len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in
           any of the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)

           Example:
           image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
           image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: will be passed to either np.pad or torch.nn.functional.pad depending on what the image is. Read the
           respective documentation!
    :param return_slicer: if True then this function will also return a tuple of python slice objects that you can use
           to crop back to the original image (reverse padding)
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
           divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match
           that will be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation (numpy) or torch.nn.functional.pad (torch)

    :returns: if return_slicer=False, this function returns the padded numpy array / torch Tensor. If
              return_slicer=True it will also return a tuple of slice objects that you can use to revert the padding:
              output, slicer = pad_nd_image(input_array, new_shape=XXX, return_slicer=True)
              reversed_padding = output[slicer] ## this is now the same as input_array, padding was reversed
    """
    if kwargs is None:
        kwargs = {}

    old_shape = np.array(image.shape)

    if shape_must_be_divisible_by is not None:
        assert isinstance(shape_must_be_divisible_by, (int, list, tuple, np.ndarray))
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (len(image.shape) - len(shape_must_be_divisible_by)) + \
                                             list(shape_must_be_divisible_by)

    if new_shape is None:
        assert shape_must_be_divisible_by is not None
        new_shape = image.shape

    if len(new_shape) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)

        if len(shape_must_be_divisible_by) < len(new_shape):
            shape_must_be_divisible_by = [1] * (len(new_shape) - len(shape_must_be_divisible_by)) + \
                                         list(shape_must_be_divisible_by)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] %
                              shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            # torch padding has the weirdest interface ever. Like wtf? Y u no read numpy documentation? So much easier
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer



def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: np.ndarray,
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False):
    predicted_logits = predicted_logits.astype(np.float32)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        return segmentation_reverted_cropping