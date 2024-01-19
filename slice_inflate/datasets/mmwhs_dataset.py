import os
import time
import glob
import json
import re
import warnings
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from joblib import Memory
import numpy as np
import einops as eo
from torch.utils.checkpoint import checkpoint

from slice_inflate.utils.common_utils import DotDict, get_script_dir
from slice_inflate.utils.torch_utils import ensure_dense, restore_sparsity, calc_dist_map, get_rotation_matrix_3d_from_angles
from slice_inflate.models.learnable_transform import get_random_affine
from slice_inflate.datasets.hybrid_id_dataset import HybridIdDataset
from slice_inflate.utils.nifti_utils import crop_around_label_center, nifti_grid_sample
from slice_inflate.utils.torch_utils import cut_slice, soft_cut_slice
from slice_inflate.datasets.clinical_cardiac_views import get_clinical_cardiac_view_affines
from slice_inflate.utils.nnunetv2_utils import get_segment_fn


cache = Memory(location=os.environ['CACHE_PATH'])
THIS_SCRIPT_DIR = get_script_dir()



class MMWHSDataset(HybridIdDataset):
    def __init__(self, *args, state='train',
                 label_tags=(
                    "background",
                    "MYO",
                    "LV",
                    "RV",
                    "LA",
                    "RA",
                ),
                 **kwargs):
        self.state = state

        if kwargs['use_2d_normal_to'] is not None:
            warnings.warn(
                "Static 2D data extraction for this dataset is skipped.")
            kwargs['use_2d_normal_to'] = None

        if kwargs['use_binarized_labels']:
            label_tags=("background", "foreground")

        super().__init__(*args, state=state, label_tags=label_tags, **kwargs)

    def extract_3d_id(self, _input):
        return _input

    def extract_short_3d_id(self, _input):
        return _input

    def __getitem__(self, dataset_id, use_2d_override=None):
        use_2d = self.use_2d(use_2d_override)
        if isinstance(dataset_id, str) and use_2d:
            dataset_idx = self.switch_2d_identifiers(dataset_id)
        elif isinstance(dataset_id, str) and not use_2d:
            dataset_idx = self.switch_3d_identifiers(dataset_id)
        else:
            dataset_idx = dataset_id

        if use_2d:
            all_ids = self.get_2d_ids()
            _id = all_ids[dataset_idx]
            image = self.img_data_2d.get(_id, torch.tensor([]))
            label = self.label_data_2d.get(_id, torch.tensor([]))

            # For 2D id cut last 4 "003rW100"
            _3d_id = self.get_3d_from_2d_identifiers(_id)
            image_path = self.img_paths.get(_3d_id)
            label_path = self.label_paths.get(_3d_id, "")

            # Additional data will only have ids for 3D samples
            additional_data = self.additional_data_3d.get(_3d_id, "")

        else:
            all_ids = self.get_3d_ids()
            _id = all_ids[dataset_idx]
            image = self.img_data_3d.get(_id, torch.tensor([]))
            label = self.label_data_3d.get(_id, torch.tensor([]))

            image_path = self.img_paths[_id]
            label_path = self.label_paths.get(_id, [])

            additional_data = self.additional_data_3d.get(_id, [])

        if self.use_modified:
            if use_2d:
                modified_label = self.modified_label_data_2d.get(
                    _id, label.detach().clone())
            else:
                modified_label = self.modified_label_data_3d.get(
                    _id, label.detach().clone())
        else:
            modified_label = label.detach().clone()

        image = image.to(device=self.device)
        label = label.to(device=self.device)

        modified_label, _ = ensure_dense(modified_label)
        modified_label = modified_label.to(device=self.device)

        if self.augment_at_collate:
            raise NotImplementedError()
            # hla_image, hla_label = image, label
            # sa_image_slc, sa_label_slc = torch.tensor([]), torch.tensor([])
            # hla_image_slc, hla_label_slc = torch.tensor([]), torch.tensor([])
            # sa_affine, hla_affine = torch.tensor([]), torch.tensor([])

        else:
            known_augment_affine = torch.eye(4)
            hidden_augment_affine = torch.eye(4)

            if self.do_augment:
                sample_augment_strength = self.self_attributes['sample_augment_strength']
                known_augment_affine = get_random_affine(
                    rotation_strength=0.,
                    zoom_strength=sample_augment_strength)

                hidden_augment_affine = get_random_affine(
                    rotation_strength=sample_augment_strength * .1,
                    zoom_strength=0.0)

            additional_data['known_augment_affine'] = known_augment_affine.view(4,4)
            additional_data['hidden_augment_affine'] = hidden_augment_affine.view(4,4)

        for key, val in additional_data.items():
            if isinstance(val, torch.Tensor):
                additional_data[key] = val.to(device=self.device)

        return dict(
            dataset_idx=dataset_idx,
            id=_id,
            image_path=image_path,
            label_path=label_path,

            image=image.to(device=self.device),
            label=label.long().to(device=self.device),

            additional_data=additional_data
        )

    @staticmethod
    def get_file_id(file_path):
        file_path = Path(file_path)
        modality, patient_id, type_str = re.findall(
            r'(ct|mr)_.*_(\d{4})_(.*?).nii.gz', file_path.name)[0]
        patient_id = int(patient_id)
        mmwhs_id = f"{modality}_{patient_id:04d}"

        is_label = type_str == 'label'
        return mmwhs_id, is_label

    @staticmethod
    def extract_2d_data(self_attributes: dict):

        # Use only specific attributes of a dict to have a cacheable function
        self = DotDict(self_attributes)

        img_data_2d = {}
        label_data_2d = {}
        modified_label_data_2d = {}

        if self.use_2d_normal_to == "D":
            slice_dim = -3
        elif self.use_2d_normal_to == "H":
            slice_dim = -2
        elif self.use_2d_normal_to == "W":
            slice_dim = -1
        elif self.use_2d_normal_to == "HLA/SA":
            pass
        else:
            raise ValueError

        if self.use_2d_normal_to:
            raise NotImplementedError()

        else:
            for _3d_id, image in self.img_data_3d.items():
                for idx, img_slc in [(slice_idx, image.select(slice_dim, slice_idx))
                                    for slice_idx in range(image.shape[slice_dim])]:
                    # Set data view for id like "003rW100"
                    img_data_2d[f"{_3d_id}:{self.use_2d_normal_to}{idx:03d}"] = img_slc

            for _3d_id, label in self.label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx))
                                    for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for id like "003rW100"
                    label_data_2d[f"{_3d_id}:{self.use_2d_normal_to}{idx:03d}"] = lbl_slc

            for _3d_id, label in self.modified_label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx))
                                    for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for id like "003rW100"
                    modified_label_data_2d[f"{_3d_id}:{self.use_2d_normal_to}{idx:03d}"] = lbl_slc

        # Postprocessing of 2d slices
        print("Postprocessing 2D slices")
        orig_2d_num = len(img_data_2d.keys())

        if self.crop_around_2d_label_center is not None:
            for _2d_id, img, label in \
                    zip(img_data_2d.keys(), img_data_2d.values(), label_data_2d.values()):

                img, label = crop_around_label_center(img, label,
                                                    torch.as_tensor(
                                                        self.crop_around_2d_label_center)
                                                    )
                img_data_2d[_2d_id] = img
                label_data_2d[_2d_id] = label

        if self.crop_2d_slices_gt_num_threshold > 0:
            for key, label in list(label_data_2d.items()):
                uniq_vals = label.unique()

                if sum(label[label > 0]) < self.crop_2d_slices_gt_num_threshold:
                    # Delete 2D slices with less than n gt-pixels (but keep 3d data)
                    del img_data_2d[key]
                    del label_data_2d[key]
                    del modified_label_data_2d[key]

        postprocessed_2d_num = len(img_data_2d.keys())
        print(
            f"Removed {orig_2d_num - postprocessed_2d_num} of {orig_2d_num} 2D slices in postprocessing")

        nonzero_lbl_percentage = torch.tensor(
            [lbl.sum((-2, -1)) > 0 for lbl in label_data_2d.values()]).sum()
        nonzero_lbl_percentage = nonzero_lbl_percentage/len(label_data_2d)
        print(f"Nonzero 2D labels: " f"{nonzero_lbl_percentage*100:.2f}%")

        nonzero_mod_lbl_percentage = torch.tensor([ensure_dense(lbl)[0].sum(
            (-2, -1)) > 0 for lbl in modified_label_data_2d.values()]).sum()
        nonzero_mod_lbl_percentage = nonzero_mod_lbl_percentage / \
            len(modified_label_data_2d)
        print(
            f"Nonzero modified 2D labels: " f"{nonzero_mod_lbl_percentage*100:.2f}%")
        print(
            f"Loader will use {postprocessed_2d_num} of {orig_2d_num} 2D slices.")

        return dict(
            img_data_2d=img_data_2d,
            label_data_2d=label_data_2d,
            modified_label_data_2d=modified_label_data_2d
        )


    @staticmethod
    def load_data(self_attributes: dict):
        # Use only specific attributes of a dict to have a cacheable function
        self = DotDict(self_attributes)

        segment_fn = get_segment_fn(self.nnunet_segment_model_path, 0, torch.device('cuda'))

        files = []
        data_path = Path(self.data_base_dir)

        # Open split json
        split_file = data_path / "metadata/data_split.json"
        with(split_file.open('r')) as split_file:
            split_dict = json.load(split_file)

        if self.crop_3d_region is not None:
            self.crop_3d_region = torch.as_tensor(self.crop_3d_region)

        files.extend(list(data_path.glob("**/*.nii.gz")))

        files = sorted(files)

        if self.state.lower() == "train":
            files = split_dict['train_files']

        elif self.state.lower() == "test":
            files = split_dict['test_files']

        elif self.state.lower() == "empty":
            state_phantoms = []
        else:
            raise Exception(
                "Unknown data state. Choose either 'train or 'test'")

        # First read filepaths
        img_paths = {}
        label_paths = {}

        if self.debug:
            files = files[:30]

        for _path in files:
            file_id, is_label = MMWHSDataset.get_file_id(_path)
            if not 'mr' in file_id: continue

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
        modified_label_data_3d = {}
        additional_data_3d = {}

        # Load data from files
        print(f"Loading MMWHS {self.state} images and labels...")
        id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

        description = f"{len(img_paths)} images, {len(label_paths)} labels"

        class_dict = {tag:idx for idx,tag in enumerate(self.label_tags)}

        for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
            additional_data_3d[_3d_id] = additional_data_3d.get(_3d_id, {})

            file_id, is_label = MMWHSDataset.get_file_id(_file)
            nib_tmp = nib.load(_file)
            tmp = torch.from_numpy(nib_tmp.get_fdata()).squeeze()
            loaded_nii_affine = torch.as_tensor(nib_tmp.affine)

            tmp, _, hires_nii_affine = nifti_grid_sample(
                tmp.unsqueeze(0).unsqueeze(0),
                loaded_nii_affine.view(1,4,4), ras_transform_mat=None,
                fov_mm=torch.as_tensor(self.hires_fov_mm), fov_vox=torch.as_tensor(self.hires_fov_vox),
                is_label=is_label,
                pre_grid_sample_affine=None,
                pre_grid_sample_hidden_affine=None,
                dtype=torch.float32
            )
            tmp = tmp[0,0]
            hires_nii_affine = hires_nii_affine[0]

            if is_label:
                if self.use_binarized_labels:
                    bin_tmp = tmp.clone()
                    bin_tmp[bin_tmp>0] = 1.0
                    label_data_3d[_3d_id] = bin_tmp.long()
                else:
                    label_data_3d[_3d_id] = tmp.long()

            else:
                if self.do_normalize:  # Normalize image to zero mean and unit std
                    tmp = (tmp - tmp.mean()) / tmp.std()
                img_data_3d[_3d_id] = tmp

            # Set additionals
            if is_label:
                additional_data_3d[_3d_id]['nifti_affine'] = hires_nii_affine # Has to be set once, either for image or label
                view_affines = get_clinical_cardiac_view_affines(
                    tmp, hires_nii_affine, class_dict,
                    num_sa_slices=15, return_unrolled=True)
                additional_data_3d[_3d_id]['gt_view_affines'] = view_affines
                # from slice_inflate.datasets.clinical_cardiac_views import display_clinical_views
                # display_clinical_views(tmp[0,0], tmp[0,0].to_sparse(), hires_nii_affine[0], view_affines,
                #     output_to_file="my_output.png")
                if self.use_distance_map_localization:
                    oh = torch.nn.functional.one_hot(tmp.long()).permute(3,0,1,2)
                    additional_data_3d[_3d_id]['label_distance_map'] = calc_dist_map(oh.unsqueeze(0).bool(), mode='outer').squeeze(0)
            else:
                if self.clinical_view_affine_type == 'from-segmented-lores-prescan':
                    # TODO improve speed for nnunet segmentation
                    lores_prescan, _, lores_nii_affine = nifti_grid_sample(
                        tmp.unsqueeze(0).unsqueeze(0),
                        hires_nii_affine.view(1,4,4), ras_transform_mat=None,
                        fov_mm=torch.as_tensor(self.lores_fov_mm), fov_vox=torch.as_tensor(self.lores_fov_vox),
                        is_label=False,
                        pre_grid_sample_affine=None,
                        pre_grid_sample_hidden_affine=None,
                        dtype=torch.float32
                    )

                    # Segment using nnunet v2 model
                    lores_spacing = torch.as_tensor(self.lores_fov_mm) / torch.as_tensor(self.lores_fov_vox)
                    lores_prescan_segmentation = segment_fn(lores_prescan.cuda(), lores_spacing.view(1,3)).cpu()

                    additional_data_3d[_3d_id]['lores_nii_affine'] = lores_nii_affine
                    additional_data_3d[_3d_id]['lores_prescan'] = lores_prescan.squeeze()
                    additional_data_3d[_3d_id]['lores_prescan_segmentation'] = lores_prescan_segmentation.squeeze()

                    additional_data_3d[_3d_id]['lores_prescan_view_affines'] = get_clinical_cardiac_view_affines(
                        lores_prescan_segmentation[0], lores_nii_affine, class_dict,
                        num_sa_slices=15, return_unrolled=True)
                    # works
                    # from slice_inflate.datasets.clinical_cardiac_views import display_clinical_views
                    # display_clinical_views(lores_prescan, lores_prescan_segmentation.to_sparse(), lores_nii_affine[0], {v:k for k,v in enumerate(self.label_tags)}, num_sa_slices=15,
                    #                         output_to_file="my_output_lores.png", debug=False)

        # Initialize 3d modified labels as unmodified labels
        for label_id in label_data_3d.keys():
            modified_label_data_3d[label_id] = label_data_3d[label_id]

        # Postprocessing of 3d volumes
        # if self.crop_around_3d_label_center is not None:
        #     for _3d_id, img, label in \
        #         zip(img_data_3d.keys(), img_data_3d.values(), label_data_3d.values()):
        #         img, label = crop_around_label_center(img, label, \
        #             torch.as_tensor(self.crop_around_3d_label_center)
        #         )
        #         img_data_3d[_3d_id] = img
        #         label_data_3d[_3d_id] = label

        return dict(
            img_paths=img_paths,
            label_paths=label_paths,
            img_data_3d=img_data_3d,
            label_data_3d=label_data_3d,
            modified_label_data_3d=modified_label_data_3d,
            additional_data_3d=additional_data_3d
        )

    @staticmethod
    def replace_label_values(label):
       raise NotImplementedError()
