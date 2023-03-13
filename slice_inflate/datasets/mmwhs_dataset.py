import os
import time
import glob
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
from slice_inflate.utils.torch_utils import ensure_dense, restore_sparsity, calc_dist_map
from slice_inflate.models.affine_transform import get_random_affine
from slice_inflate.datasets.hybrid_id_dataset import HybridIdDataset
from slice_inflate.datasets.align_mmwhs import crop_around_label_center, cut_slice, soft_cut_slice, nifti_transform

cache = Memory(location=os.environ['MMWHS_CACHE_PATH'])
THIS_SCRIPT_DIR = get_script_dir()



class MMWHSDataset(HybridIdDataset):
    def __init__(self, *args, state='train',
                 label_tags=(
            "background",
            "left_myocardium",
            "left_atrium",
            "left_ventricle",
            "right_atrium",
            "right_ventricle",
            # "ascending_aorta", # This label is currently purged from the data
            # "pulmonary_artery" # This label is currently purged from the data
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
        # Match sth like 1001-HLA:mTS1
        items = re.findall(r'^(\d{4})-(ct|mr)(:m[A-Z0-9a-z]{3,4})?', _input)[0]
        items = list(filter(None, items))
        return "-".join(items)

    def extract_short_3d_id(self, _input):
        # Match sth like 1001-HLA:mTS1 and returns 1001-HLA
        items = re.findall(r'^(\d{4})-(HLA)', _input)[0]
        items = list(filter(None, items))
        return "-".join(items)

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

        augment_affine = None

        if self.augment_at_collate:
            raise NotImplementedError()
            # hla_image, hla_label = image, label
            # sa_image_slc, sa_label_slc = torch.tensor([]), torch.tensor([])
            # hla_image_slc, hla_label_slc = torch.tensor([]), torch.tensor([])
            # sa_affine, hla_affine = torch.tensor([]), torch.tensor([])

        else:
            augment_affine = torch.eye(4)

            if self.do_augment:
                sample_augment_strength = self.self_attributes['sample_augment_strength']
                augment_affine = get_random_affine(sample_augment_strength)

            additional_data['augment_affine'] = augment_affine.view(4,4)

            D, H, W = label.shape

        additional_data['label_distance_map'] = additional_data['label_distance_map'].to(device=self.device)

        return dict(
            dataset_idx=dataset_idx,
            id=_id,
            image_path=image_path,
            label_path=label_path,

            image=image.to(device=self.device),
            label=label.long().to(device=self.device),

            additional_data=additional_data
        )

    def get_efficient_augmentation_collate_fn(self):

        def collate_closure(batch):
            batch = torch.utils.data._utils.collate.default_collate(batch)
            if self.augment_at_collate:
                B = batch['dataset_idx'].shape[0]

                image = batch['image']
                label = batch['label']
                additional_data = batch['additional_data']

                all_hla_images = []
                all_hla_labels = []
                all_sa_image_slcs = []
                all_sa_label_slcs = []
                all_hla_image_slcs = []
                all_hla_label_slcs = []
                all_sa_affines = []
                all_hla_affines = []

                B, D, H, W = batch['label'].shape

                image = batch['image'].cuda()
                label = batch['label'].view(B, 1, D, H, W).cuda()

                nifti_affine = additional_data['nifti_affine'].to(
                    device=label.device).view(B, 4, 4)
                augment_affine = torch.eye(4).view(1, 4, 4).repeat(
                    B, 1, 1).to(device=label.device)

                if self.do_augment:
                    for b_idx in range(B):
                        augment_angle_std = self.self_attributes['augment_angle_std']
                        deg_angles = torch.normal(
                            mean=0, std=augment_angle_std*torch.ones(3))
                        augment_affine[b_idx, :3, :3] = get_rotation_matrix_3d_from_angles(
                            deg_angles)

                sa_image, sa_label, sa_image_slc, sa_label_slc, sa_affine = \
                    self.get_transformed(
                        label, nifti_affine, augment_affine, 'sa', image)
                hla_image, hla_label, hla_image_slc, hla_label_slc, hla_affine = \
                    self.get_transformed(
                        label, nifti_affine, augment_affine, 'hla', image)

                all_hla_images.append(hla_image)
                all_hla_labels.append(hla_label)
                all_sa_image_slcs.append(sa_image_slc)
                all_sa_label_slcs.append(sa_label_slc)
                all_hla_image_slcs.append(hla_image_slc)
                all_hla_label_slcs.append(hla_label_slc)
                all_sa_affines.append(sa_affine)
                all_hla_affines.append(hla_affine)

                batch.update(dict(
                    image=torch.cat(all_hla_images, dim=0),
                    label=torch.cat(all_hla_labels, dim=0),

                    sa_image_slc=torch.cat(all_sa_image_slcs, dim=0),
                    sa_label_slc=torch.cat(all_sa_label_slcs, dim=0),

                    hla_image_slc=torch.cat(all_hla_image_slcs, dim=0),
                    hla_label_slc=torch.cat(all_hla_label_slcs, dim=0),

                    sa_affine=torch.stack(all_sa_affines),
                    hla_affine=torch.stack(all_hla_affines)
                ))

            return batch

        return collate_closure



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
                img_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = img_slc

        for _3d_id, label in self.label_data_3d.items():
            for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx))
                                 for slice_idx in range(label.shape[slice_dim])]:
                # Set data view for id like "003rW100"
                label_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = lbl_slc

        for _3d_id, label in self.modified_label_data_3d.items():
            for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx))
                                 for slice_idx in range(label.shape[slice_dim])]:
                # Set data view for id like "003rW100"
                modified_label_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = lbl_slc

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



def load_data(self_attributes: dict):
    # Use only specific attributes of a dict to have a cacheable function
    self = DotDict(self_attributes)

    IMAGE_ID = '_image'

    t0 = time.time()

    if self.modality == 'all':
        modalities = ['mr', 'ct']
    else:
        modalities = [self.modality]

    files = []

    for mod in modalities:
        if self.state.lower() == "train":
            data_directory = f"{mod}_registered_train_selection"

        elif self.state.lower() == "test":
            data_directory = f"{mod}_registered_test_selection"

        elif self.state.lower() == "empty":
            data_directory = "nonexisting_dir_4t6yh"
        else:
            raise Exception(
                "Unknown data state. Choose either 'train or 'test'")

        data_path = Path(self.base_dir, data_directory)

        if self.crop_3d_region is not None:
            self.crop_3d_region = torch.as_tensor(self.crop_3d_region)

        files.extend(list(data_path.glob("**/*.nii.gz")))

    files = sorted(files)

    # First read filepaths
    img_paths = {}
    label_paths = {}

    if self.debug:
        files = files[:2]

    for _path in files:
        trailing_name = str(_path).split("/")[-1]
        modality, patient_id = re.findall(
            r'(ct|mr)_.*_(\d{4})_.*?.nii.gz', trailing_name)[0]
        patient_id = int(patient_id)

        # Generate cmrxmotion id like 001-02-ES
        mmwhs_id = f"{patient_id:04d}-{modality}"

        if not IMAGE_ID in trailing_name:
            label_paths[mmwhs_id] = str(_path)
        else:
            img_paths[mmwhs_id] = str(_path)

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
    print(f"Loading MMWHS {self.state} images and labels... ({modalities})")
    id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

    description = f"{len(img_paths)} images, {len(label_paths)} labels"

    for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
        additional_data_3d[_3d_id] = additional_data_3d.get(_3d_id, {})

        trailing_name = str(_file).split("/")[-1]
        is_label = not IMAGE_ID in trailing_name
        nib_tmp = nib.load(_file)
        tmp = torch.from_numpy(nib_tmp.get_fdata()).squeeze()

        align_affine_path = str(Path(self.base_dir, "preprocessed",
                                f"f1002mr_m{_3d_id.split('-')[0]}{_3d_id.split('-')[1]}.mat"))
        augment_affine = torch.from_numpy(np.loadtxt(align_affine_path))
        affine = torch.as_tensor(nib_tmp.affine)

        additional_data_3d[_3d_id]['nifti_affine'] = affine

        if is_label:
            tmp = replace_label_values(tmp)

            if self.use_binarized_labels:
                tmp[tmp>0] = 1.0

        tmp, _, affine = nifti_transform(
            tmp.unsqueeze(0).unsqueeze(0),
            affine.view(1,4,4), ras_transform_mat=None,
            fov_mm=torch.as_tensor(self.fov_mm), fov_vox=torch.as_tensor(self.fov_vox),
            is_label=is_label,
            pre_grid_sample_affine=None,
            pre_grid_sample_augment_affine=None,
            dtype=torch.float32
        )
        tmp = tmp.squeeze(0).squeeze(0)
        affine = affine.squeeze(0)

        if not IMAGE_ID in trailing_name:
            label_data_3d[_3d_id] = tmp.long()
            oh = torch.nn.functional.one_hot(tmp.long()).permute(3,0,1,2)
            additional_data_3d[_3d_id]['label_distance_map'] = calc_dist_map(oh.unsqueeze(0).bool(), mode='outer').squeeze(0)

        else:
            if self.do_normalize:  # Normalize image to zero mean and unit std
                tmp = (tmp - tmp.mean()) / tmp.std()

            img_data_3d[_3d_id] = tmp

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


def replace_label_values(label):
    # Replace label numbers with MMWHS equivalent
    # STRUCTURE           MMWHS   ACDC    NNUNET
    # background          0       0       0
    # left_myocardium     205     2       1
    # left_atrium         420     N/A     2
    # ?                   421     N/A     N/A
    # left_ventricle      500     3       3
    # right_atrium        550     N/A     4
    # right_ventricle     600     1       5
    # ascending_aorta     820     N/A     6
    # pulmonary_artery    850     N/A     7
    orig_values = [0,  205, 420, 421, 500, 550, 600, 820, 850]
    new_values = [0,  1,   2,   0,   3,   4,   5,   0,   0]

    modified_label = label.clone()
    for orig, new in zip(orig_values, new_values):
        modified_label[modified_label == orig] = new
    return modified_label
