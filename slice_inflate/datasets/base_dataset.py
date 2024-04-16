import os
from pathlib import Path
from abc import abstractmethod
from collections.abc import Iterable
from collections import OrderedDict
import json

from tqdm import tqdm
from joblib import Memory
import einops as eo

import numpy as np
import torch
from torch.utils.data import Dataset
import monai
import nibabel as nib

from slice_inflate.utils.torch_utils import ensure_dense, ensure_dense, get_batch_score_per_label, reduce_label_scores_epoch
from slice_inflate.utils.register_centroids import get_centroid_reorient_grid_affine
from slice_inflate.utils.nnunetv2_utils import get_segment_fn
from slice_inflate.utils.python_utils import get_script_dir
from slice_inflate.utils.nifti_utils import nifti_grid_sample, get_zooms
from slice_inflate.utils.clinical_cardiac_views import get_clinical_cardiac_view_affines
from slice_inflate.utils.log_utils import log_oa_metrics, log_label_metrics

THIS_SCRIPT_DIR = get_script_dir()

class BaseDataset(Dataset):

    def __init__(self,
        data_base_dir,
        ensure_labeled_pairs=True,
        do_normalize:bool=True,
        label_tags=(),
        device='cpu', debug=False,
        **kwargs
    ):

        # Prepare an attribute dict to identify all dataset settings for joblib
        self.self_attributes = locals().copy()
        for arg_name, arg_value in kwargs.items():
            self.self_attributes[arg_name] = arg_value

        del self.self_attributes['kwargs']
        del self.self_attributes['self']

        self.segment_fn = get_segment_fn(self.self_attributes['nnunet_segment_model_path'], 0, torch.device('cuda'))
        with open(Path(data_base_dir) / "metadata/data_split.json", 'r') as f:
            self.data_split = json.load(f)
        self.data_base_dir = data_base_dir
        self.ensure_labeled_pairs = ensure_labeled_pairs
        self.do_normalize = do_normalize
        self.device = device
        self.debug = debug

        self.label_tags = label_tags
        self.disturbed_idxs = []

        # Load base 3D data
        all_3d_data_dict = self.load_data()

        self.self_attributes['img_paths'] = self.img_paths = all_3d_data_dict.pop('img_paths', {})
        self.self_attributes['label_paths'] = self.label_paths = all_3d_data_dict.pop('label_paths', {})
        self.self_attributes['img_data_3d'] = self.img_data_3d = all_3d_data_dict.pop('img_data_3d', {})
        self.self_attributes['label_data_3d'] = self.label_data_3d = all_3d_data_dict.pop('label_data_3d', {})
        self.self_attributes['additional_data_3d'] = self.additional_data_3d = all_3d_data_dict.pop('additional_data_3d', {})

        # Check for consistency
        print(f"Equal image and label numbers: {set(self.img_data_3d)==set(self.label_data_3d)} ({len(self.img_data_3d)})")

        # Now make sure dicts are ordered
        self.img_paths = OrderedDict(sorted(self.img_paths.items()))
        self.label_paths = OrderedDict(sorted(self.label_paths.items()))
        self.img_data_3d = OrderedDict(sorted(self.img_data_3d.items()))
        self.label_data_3d = OrderedDict(sorted(self.label_data_3d.items()))

        print("Data import finished.")

    def get_3d_ids(self):
        return list(self.img_data_3d.keys())

    def switch_3d_identifiers(self, _3d_identifiers):
        if isinstance(_3d_identifiers, (torch.Tensor, np.ndarray)):
            _3d_identifiers = _3d_identifiers.tolist()
        elif not isinstance(_3d_identifiers, Iterable) or isinstance(_3d_identifiers, str):
            _3d_identifiers = [_3d_identifiers]

        _ids = self.get_3d_ids()
        if all([isinstance(elem, int) for elem in _3d_identifiers]):
            vals = [_ids[elem] for elem in _3d_identifiers]
        elif all([isinstance(elem, str) for elem in _3d_identifiers]):
            vals = [_ids.index(elem) if elem in _ids else None for elem in _3d_identifiers]
        else:
            raise ValueError
        return vals

    def __len__(self):
        return len(self.img_data_3d)

    def __getitem__(self, dataset_id, use_2d_override=None):
        if isinstance(dataset_id, str):
            dataset_idx = self.switch_3d_identifiers(dataset_id)
        else:
            dataset_idx = dataset_id

        all_ids = self.get_3d_ids()
        _id = all_ids[dataset_idx]
        image = self.img_data_3d.get(_id, torch.tensor([]))
        label = self.label_data_3d.get(_id, torch.tensor([]))

        image_path = self.img_paths[_id]
        label_path = self.label_paths.get(_id, [])

        additional_data = self.additional_data_3d.get(_id, [])

        image = image.to(device=self.device)
        label = label.to(device=self.device)

        for key, val in additional_data.items():
            if isinstance(val, torch.Tensor):
                additional_data[key] = val.to(device=self.device)

        return dict(
            dataset_idx=dataset_idx,
            id=_id,
            image_path=image_path,
            label_path=label_path,

            image=image.to(device=self.device),
            label=ensure_dense(label).long().to(device=self.device),

            additional_data=additional_data
        )

    def load_data(self):
        segment_fn = get_segment_fn(self.nnunet_segment_model_path, 0, torch.device('cuda'))

        files = []
        data_path = Path(self.data_base_dir)

        # Open split json
        split_file = data_path / "metadata/data_split.json"
        with(split_file.open('r')) as split_file:
            split_dict = json.load(split_file)

        if self.state.lower() == "train":
            files = split_dict['train_files']

        elif self.state.lower() == "test":
            files = split_dict['test_files']

        elif self.state.lower() == "empty":
            files = []

        else:
            raise Exception(
                "Unknown data state. Choose either 'train or 'test' or 'empty'")

        # First read filepaths
        img_paths = {}
        label_paths = {}

        if self.debug:
            files = files[:30]

        for _path in files:
            file_id, is_label = self.get_file_id(_path)

            if is_label:
                label_paths[file_id] = str(_path)
            else:
                img_paths[file_id] = str(_path)

        assert len(img_paths) > 0

        if self.ensure_labeled_pairs:
            pair_idxs = set(img_paths).intersection(set(label_paths))
            label_paths = {_id: _path for _id,
                        _path in label_paths.items() if _id in pair_idxs}
            img_paths = {_id: _path for _id,
                        _path in img_paths.items() if _id in pair_idxs}

        img_data_3d = {}
        label_data_3d = {}
        additional_data_3d = {}

        # Load data from files
        print(f"Loading {self.state} images and labels...")
        id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

        description = f"{len(img_paths)} images, {len(label_paths)} labels"

        class_dict = {tag:idx for idx,tag in enumerate(self.label_tags)}

        label_scores_dataset = {}
        fixed_ref_path = Path(THIS_SCRIPT_DIR, './ref_heart.nii.gz')

        for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
            additional_data_3d[_3d_id] = additional_data_3d.get(_3d_id, {})

            file_id, is_label = self.get_file_id(_file)
            nib_tmp = nib.load(_file)
            tmp = torch.from_numpy(nib_tmp.get_fdata()).squeeze()
            if not is_label:
                tmp = tmp.float()
            loaded_nii_affine = torch.as_tensor(nib_tmp.affine)

            tmp, _, hires_nii_affine = nifti_grid_sample(
                tmp.unsqueeze(0).unsqueeze(0),
                loaded_nii_affine.view(1,4,4), ras_transform_affine=None,
                target_fov_mm=torch.as_tensor(self.self_attributes['hires_fov_mm']), target_fov_vox=torch.as_tensor(self.self_attributes['hires_fov_vox']),
                is_label=is_label,
                pre_grid_sample_affine=None,
                # pre_grid_sample_hidden_affine=None,
                dtype=torch.float32
            )
            tmp = tmp[0,0]
            hires_nii_affine = hires_nii_affine[0]

            if is_label:
                label_data_3d[_3d_id] = tmp.long()

            else:
                if self.do_normalize:  # Normalize image to zero mean and unit std
                    tmp = (tmp - tmp.mean()) / tmp.std()
                img_data_3d[_3d_id] = tmp

            # Set additionals
            if is_label:
                additional_data_3d[_3d_id]['nifti_affine'] = hires_nii_affine # Has to be set once, either for image or label
                # print(get_zooms(hires_nii_affine[None]))
                view_affines = get_clinical_cardiac_view_affines(
                    tmp, hires_nii_affine, class_dict,
                    num_sa_slices=15, return_unrolled=True)

                centroids_affine = get_centroid_reorient_grid_affine(tmp.int(), fixed_ref_path)
                view_affines['centroids'] = centroids_affine

                additional_data_3d[_3d_id]['gt_view_affines'] = view_affines
                # from slice_inflate.datasets.clinical_cardiac_views import display_clinical_views
                # display_clinical_views(tmp[0,0], tmp[0,0].to_sparse(), hires_nii_affine[0], view_affines,
                #     output_to_file="my_output.png")

                # Save prescan gt
                prescan_label, _, prescan_nii_affine = nifti_grid_sample(
                    tmp.unsqueeze(0).unsqueeze(0),
                    hires_nii_affine.view(1,4,4), ras_transform_affine=None,
                    target_fov_mm=torch.as_tensor(self.self_attributes['prescan_fov_mm']), target_fov_vox=torch.as_tensor(self.self_attributes['prescan_fov_vox']),
                    is_label=True,
                    pre_grid_sample_affine=None,
                    # pre_grid_sample_hidden_affine=None,
                    dtype=torch.float32
                )
                prescan_label = prescan_label.long()

                # Segment using nnunet v2 model
                lores_spacing = get_zooms(prescan_nii_affine)
                additional_data_3d[_3d_id]['prescan_nii_affine'] = prescan_nii_affine.squeeze()
                additional_data_3d[_3d_id]['prescan_gt'] = prescan_label.squeeze()

            if not is_label and self.self_attributes['clinical_view_affine_type'] == 'from-segmented':
                # Segment from image
                prescan_image, _, prescan_nii_affine = nifti_grid_sample(
                    tmp.unsqueeze(0).unsqueeze(0),
                    hires_nii_affine.view(1,4,4), ras_transform_affine=None,
                    target_fov_mm=torch.as_tensor(self.self_attributes['prescan_fov_mm']), target_fov_vox=torch.as_tensor(self.self_attributes['prescan_fov_vox']),
                    is_label=False,
                    pre_grid_sample_affine=None,
                    # pre_grid_sample_hidden_affine=None,
                    dtype=torch.float32
                )

                # Segment using nnunet v2 model
                lores_spacing = get_zooms(prescan_nii_affine)
                prescan_segmentation = segment_fn(prescan_image.cuda(), lores_spacing.view(1,3)).cpu()

                additional_data_3d[_3d_id]['prescan_image'] = prescan_image.squeeze()
                additional_data_3d[_3d_id]['prescan_label'] = prescan_segmentation.squeeze()

                additional_data_3d[_3d_id]['prescan_view_affines'] = get_clinical_cardiac_view_affines(
                    additional_data_3d[_3d_id]['prescan_label'], additional_data_3d[_3d_id]['prescan_nii_affine'], class_dict,
                    num_sa_slices=15, return_unrolled=True)

                prescan_centroids_affine = get_centroid_reorient_grid_affine(tmp.int(), fixed_ref_path, DOF=6)
                additional_data_3d[_3d_id]['prescan_view_affines']['centroids'] = prescan_centroids_affine
                # works
                # from slice_inflate.datasets.clinical_cardiac_views import display_clinical_views
                # display_clinical_views(prescan, prescan_segmentation.to_sparse(), prescan_nii_affine[0], {v:k for k,v in enumerate(self.label_tags)}, num_sa_slices=15,
                #                         output_to_file="my_output_lores.png", debug=False)

                # Calculate dice score
                prescan_seg_interp = torch.nn.functional.interpolate(prescan_segmentation[None], size=tmp.shape, mode='nearest')[0]
                pred_prescan_seg_oh = eo.rearrange(torch.nn.functional.one_hot(prescan_seg_interp.long(), len(self.label_tags)), 'b d h w oh -> b oh d h w')
                target_oh = eo.rearrange(torch.nn.functional.one_hot(label_data_3d[_3d_id][None].long(), len(self.label_tags)), 'b d h w oh -> b oh d h w')
                case_dice = monai.metrics.compute_dice(pred_prescan_seg_oh, label_data_3d[_3d_id][None,None].long())
                label_scores_dataset = get_batch_score_per_label(label_scores_dataset, 'dice',
                    case_dice, self.label_tags, exclude_bg=True)

                case_hd95 = monai.metrics.compute_hausdorff_distance(pred_prescan_seg_oh, target_oh, percentile=95) * get_zooms(loaded_nii_affine[None]).norm()
                case_hd95 = torch.cat([torch.zeros(1,1).to(case_hd95), case_hd95], dim=1) # Add zero score for background
                label_scores_dataset = get_batch_score_per_label(label_scores_dataset, 'hd95',
                    case_hd95, self.label_tags, exclude_bg=True)

            elif is_label and self.self_attributes['clinical_view_affine_type'] == 'from-gt':
                # Take GT for prescan
                additional_data_3d[_3d_id]['prescan_label'] = additional_data_3d[_3d_id]['prescan_gt']

                additional_data_3d[_3d_id]['prescan_view_affines'] = get_clinical_cardiac_view_affines(
                    additional_data_3d[_3d_id]['prescan_label'], additional_data_3d[_3d_id]['prescan_nii_affine'], class_dict,
                    num_sa_slices=15, return_unrolled=True)
                # works
                # from slice_inflate.datasets.clinical_cardiac_views import display_clinical_views
                # display_clinical_views(prescan, prescan_segmentation.to_sparse(), prescan_nii_affine[0], {v:k for k,v in enumerate(self.label_tags)}, num_sa_slices=15,
                #                         output_to_file="my_output_lores.png", debug=False)

        seg_metrics_nanmean_per_label, seg_metrics_std_per_label, seg_metrics_nanmean_oa, seg_metrics_std_oa  = reduce_label_scores_epoch(label_scores_dataset)
        log_label_metrics(f"dataset/prescan_mean", '', seg_metrics_nanmean_per_label, 0,
            logger_selected_metrics=(), print_selected_metrics=('dice', 'hd95'))
        log_label_metrics(f"dataset/prescan_mean", '', seg_metrics_std_per_label, 0,
            logger_selected_metrics=(), print_selected_metrics=('dice', 'hd95'))
        log_oa_metrics(f"dataset/prescan_mean_oa_exclude_bg", '', seg_metrics_nanmean_oa, 0,
            logger_selected_metrics=(), print_selected_metrics=('dice', 'hd95'))
        log_oa_metrics(f"dataset/prescan_mean_oa_exclude_bg", '', seg_metrics_std_oa, 0,
            logger_selected_metrics=(), print_selected_metrics=('dice', 'hd95'))
        print()

        return dict(
            img_paths=img_paths,
            label_paths=label_paths,
            img_data_3d=img_data_3d,
            label_data_3d=label_data_3d,
            additional_data_3d=additional_data_3d
        )

    @abstractmethod
    def extract_3d_id(self, _input):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_file_id( file_path):
        raise NotImplementedError()