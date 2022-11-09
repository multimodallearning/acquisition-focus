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
from slice_inflate.utils.torch_utils import ensure_dense, restore_sparsity, get_rotation_matrix_3d_from_angles
from slice_inflate.datasets.hybrid_id_dataset import HybridIdDataset
from slice_inflate.datasets.align_mmwhs import crop_around_label_center, cut_slice, soft_cut_slice, nifti_transform

cache = Memory(location=os.environ['MMWHS_CACHE_PATH'])
THIS_SCRIPT_DIR = get_script_dir()

class AffineTransformModule(torch.nn.Module):
    def __init__(self, fov_mm, fov_vox, view_affine, do_transform_images=False, do_transform_labels=False):
        super().__init__()

        self.fov_mm = fov_mm
        self.fov_vox = fov_vox
        self.do_transform_images = do_transform_images
        self.do_transform_labels = do_transform_labels
        self.fov_mm = fov_mm
        self.fov_vox = fov_vox
        self.view_affine = view_affine
        self.theta_t = torch.nn.Parameter(torch.zeros(3))
        self.theta_m = torch.nn.Parameter(torch.eye(3))

    def get_batch_affine(self, batch_size):
        # theta = torch.cat([self.theta_m, self.theta_t.view(3,1)], dim=1)
        theta = torch.cat([torch.eye(3, device=self.theta_t.device), self.theta_t.view(3,1)], dim=1)
        theta = torch.cat([theta, torch.tensor([0,0,0,1], device=theta.device).view(1,4)], dim=0)
        return theta.view(1,4,4).repeat(batch_size,1,1)

    def forward(self, x_image, x_label, nifti_affine, align_affine):
        y_image, y_label, affine = x_image, x_label, None
        # Compose the complete alignment from:
        # 1) view (sa, hla)
        # 2) learnt affine
        # 3) and align affine (global alignment @ augment, if any)

        B = y_label.shape[0]
        final_align_affine = (
            self.view_affine.to(x_label.device)
            @ self.get_batch_affine(B).to(x_label.device)
            @ align_affine.to(x_label.device)
        )

        if self.do_transform_images:
            y_image, affine = nifti_transform(x_image, nifti_affine, final_align_affine,
                fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=False)

        if self.do_transform_labels:
            y_label, affine = nifti_transform(x_label, nifti_affine, final_align_affine,
                fov_mm=self.fov_mm, fov_vox=self.fov_vox, is_label=True)

        return y_image, y_label, affine



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
        self.io_normalisation_values = torch.load(Path(args[0], "mmwhs_io_normalisation_values.pth"))

        if kwargs['use_2d_normal_to'] is not None:
            warnings.warn("Static 2D data extraction for this dataset is skipped.")
            kwargs['use_2d_normal_to'] = None

        hla_affine_path = Path(
            THIS_SCRIPT_DIR,
            "slice_inflate/preprocessing",
            "mmwhs_1002_HLA_red_slice_to_ras.mat"
        )
        sa_affine_path =  Path(
            THIS_SCRIPT_DIR,
            "slice_inflate/preprocessing",
            "mmwhs_1002_SA_yellow_slice_to_ras.mat"
        )

        self.sa_atm = AffineTransformModule(
            torch.tensor(kwargs['fov_mm']),
            torch.tensor(kwargs['fov_vox']),
            view_affine=torch.as_tensor(np.loadtxt(sa_affine_path)).float(),
            do_transform_images=True, do_transform_labels=True)

        self.hla_atm = AffineTransformModule(
            torch.tensor(kwargs['fov_mm']),
            torch.tensor(kwargs['fov_vox']),
            view_affine=torch.as_tensor(np.loadtxt(hla_affine_path)).float(),
            do_transform_images=True, do_transform_labels=True)

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
                modified_label = self.modified_label_data_2d.get(_id, label.detach().clone())
            else:
                modified_label = self.modified_label_data_3d.get(_id, label.detach().clone())
        else:
            modified_label = label.detach().clone()

        image = image.to(device=self.device)
        label = label.to(device=self.device)

        modified_label, _ = ensure_dense(modified_label)
        modified_label = modified_label.to(device=self.device)

        augment_affine = None

        if self.augment_at_collate:
            sa_image, sa_label = image, label
            sa_image_slc, sa_label_slc = torch.tensor([]), torch.tensor([])
            hla_image_slc, hla_label_slc = torch.tensor([]), torch.tensor([])
            sa_affine, hla_affine = torch.tensor([]), torch.tensor([])

        else:
            augment_affine = torch.eye(4)

            if self.do_augment:
                augment_angle_std = self.self_attributes['augment_angle_std']
                deg_angles = torch.normal(mean=0, std=augment_angle_std*torch.ones(3))
                augment_affine[:3,:3] = get_rotation_matrix_3d_from_angles(deg_angles)

            nifti_affine = additional_data['nifti_affine'].view(1,4,4)
            augment_affine = augment_affine.view(1,4,4)

            D,H,W = label.shape

            sa_image, sa_label, sa_image_slc, sa_label_slc, sa_affine = \
                self.get_transformed(label.view(1,1,D,H,W), nifti_affine, augment_affine, 'sa', image=None)
            hla_image, hla_label, hla_image_slc, hla_label_slc, hla_affine = \
                self.get_transformed(label.view(1,1,D,H,W), nifti_affine, augment_affine, 'hla', image=None)

            sa_image, sa_label, sa_image_slc, sa_label_slc, sa_affine = \
                sa_image.squeeze(0), sa_label.squeeze(0), sa_image_slc.squeeze(0), sa_label_slc.squeeze(0), sa_affine.squeeze(0)
            hla_image, hla_label, hla_image_slc, hla_label_slc, hla_affine = \
                hla_image.squeeze(0), hla_label.squeeze(0), hla_image_slc.squeeze(0), hla_label_slc.squeeze(0), hla_affine.squeeze(0)

        return dict(
            dataset_idx=dataset_idx,
            id=_id,
            image_path=image_path,
            label_path=label_path,

            image=sa_image.to(device=self.device),
            sa_image_slc=sa_image_slc.to(device=self.device),
            hla_image_slc=hla_image_slc.to(device=self.device),

            label=sa_label.long().to(device=self.device),
            sa_label_slc=sa_label_slc.long().to(device=self.device),
            hla_label_slc=hla_label_slc.long().to(device=self.device),

            sa_affine=sa_affine,
            hla_affine=hla_affine,

            additional_data=additional_data
        )

    def get_transformed(self, label, nifti_affine, align_affine, atm_name, image=None):

        assert atm_name in ['sa', 'hla']
        img_is_invalid = image is None or image.dim() == 0
        if img_is_invalid:
            image = torch.zeros_like(label)

        B,C,D,H,W = label.shape
        CLASS_NUM = len(self.self_attributes['label_tags'])
        label = eo.rearrange(F.one_hot(label, CLASS_NUM), 'b c d h w oh -> b (c oh) d h w')

        if atm_name == 'sa':
            atm = self.sa_atm
        elif atm_name == 'hla':
            atm = self.hla_atm


       # Transform label with 'bilinear' interpolation to have gradients
        soft_label, _, _ = atm(label.float().view(B,CLASS_NUM,D,H,W), label.view(B,CLASS_NUM,D,H,W), nifti_affine, align_affine)

        image, label, affine = atm(image.view(B,C,D,H,W), label.view(B,CLASS_NUM,D,H,W), nifti_affine, align_affine)

        if self.self_attributes['crop_around_3d_label_center'] is not None:
            _3d_vox_size = torch.as_tensor(self.self_attributes['crop_around_3d_label_center'])
            label, image, _ = crop_around_label_center(label, _3d_vox_size, image)
            _, soft_label, _ = crop_around_label_center(label, _3d_vox_size, soft_label)

        label_slc = soft_cut_slice(soft_label)
        image_slc = cut_slice(image)

        if self.self_attributes['crop_around_2d_label_center'] is not None:
            _2d_vox_size = torch.as_tensor(self.self_attributes['crop_around_2d_label_center']+[1])
            label_slc, image_slc, _ = crop_around_label_center(label_slc, _2d_vox_size, image_slc)

        if img_is_invalid:
            image = torch.empty([])
            image_slc = torch.empty([])
            # Do not set label_slc to .int() here, since we (may) need the gradients
        return image, label.int(), image_slc, label_slc, affine.float()

    def get_efficient_augmentation_collate_fn(self):

        def collate_closure(batch):
            batch = torch.utils.data._utils.collate.default_collate(batch)
            if self.augment_at_collate:
                B = batch['dataset_idx'].shape[0]

                image = batch['image']
                label = batch['label']
                additional_data = batch['additional_data']

                all_sa_images = []
                all_sa_labels = []
                all_sa_image_slcs = []
                all_sa_label_slcs = []
                all_hla_image_slcs = []
                all_hla_label_slcs = []
                all_sa_affines = []
                all_hla_affines = []

                B,D,H,W = batch['label'].shape

                image = batch['image'].cuda()
                label = batch['label'].view(B,1,D,H,W).cuda()

                nifti_affine = additional_data['nifti_affine'].to(device=label.device).view(B,4,4)
                augment_affine = torch.eye(4).view(1,4,4).repeat(B,1,1).to(device=label.device)

                if self.do_augment:
                    for b_idx in range(B):
                        augment_angle_std = self.self_attributes['augment_angle_std']
                        deg_angles = torch.normal(mean=0, std=augment_angle_std*torch.ones(3))
                        augment_affine[b_idx,:3,:3] = get_rotation_matrix_3d_from_angles(deg_angles)

                sa_image, sa_label, sa_image_slc, sa_label_slc, sa_affine = \
                    self.get_transformed(label, nifti_affine, augment_affine, 'sa', image)
                hla_image, hla_label, hla_image_slc, hla_label_slc, hla_affine = \
                    self.get_transformed(label, nifti_affine, augment_affine, 'hla', image)

                all_sa_images.append(sa_image)
                all_sa_labels.append(sa_label)
                all_sa_image_slcs.append(sa_image_slc)
                all_sa_label_slcs.append(sa_label_slc)
                all_hla_image_slcs.append(hla_image_slc)
                all_hla_label_slcs.append(hla_label_slc)
                all_sa_affines.append(sa_affine)
                all_hla_affines.append(hla_affine)

                batch.update(dict(
                    image=torch.cat(all_sa_images, dim=0),
                    label=torch.cat(all_sa_labels, dim=0),

                    sa_image_slc=torch.cat(all_sa_image_slcs, dim=0),
                    sa_label_slc=torch.cat(all_sa_label_slcs, dim=0),

                    hla_image_slc=torch.cat(all_hla_image_slcs, dim=0),
                    hla_label_slc=torch.cat(all_hla_label_slcs, dim=0),

                    sa_affine=torch.stack(all_sa_affines),
                    hla_affine=torch.stack(all_hla_affines)
                ))

            return batch

        return collate_closure



# @cache.cache(verbose=True)
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
            for idx, img_slc in [(slice_idx, image.select(slice_dim, slice_idx)) \
                                    for slice_idx in range(image.shape[slice_dim])]:
                # Set data view for id like "003rW100"
                img_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = img_slc

        for _3d_id, label in self.label_data_3d.items():
            for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                    for slice_idx in range(label.shape[slice_dim])]:
                # Set data view for id like "003rW100"
                label_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = lbl_slc

        for _3d_id, label in self.modified_label_data_3d.items():
            for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                    for slice_idx in range(label.shape[slice_dim])]:
                # Set data view for id like "003rW100"
                modified_label_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = lbl_slc


    # Postprocessing of 2d slices
    print("Postprocessing 2D slices")
    orig_2d_num = len(img_data_2d.keys())

    if self.crop_around_2d_label_center is not None:
        for _2d_id, img, label in \
            zip(img_data_2d.keys(), img_data_2d.values(), label_data_2d.values()):

            img, label = crop_around_label_center(img, label, \
                torch.as_tensor(self.crop_around_2d_label_center)
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
    print(f"Removed {orig_2d_num - postprocessed_2d_num} of {orig_2d_num} 2D slices in postprocessing")

    nonzero_lbl_percentage = torch.tensor([lbl.sum((-2,-1)) > 0 for lbl in label_data_2d.values()]).sum()
    nonzero_lbl_percentage = nonzero_lbl_percentage/len(label_data_2d)
    print(f"Nonzero 2D labels: " f"{nonzero_lbl_percentage*100:.2f}%")

    nonzero_mod_lbl_percentage = torch.tensor([ensure_dense(lbl)[0].sum((-2,-1)) > 0 for lbl in modified_label_data_2d.values()]).sum()
    nonzero_mod_lbl_percentage = nonzero_mod_lbl_percentage/len(modified_label_data_2d)
    print(f"Nonzero modified 2D labels: " f"{nonzero_mod_lbl_percentage*100:.2f}%")
    print(f"Loader will use {postprocessed_2d_num} of {orig_2d_num} 2D slices.")

    return dict(
        img_data_2d=img_data_2d,
        label_data_2d=label_data_2d,
        modified_label_data_2d=modified_label_data_2d
    )



# @cache.cache(verbose=True)
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
            data_directory = f"{mod}_registered_train"

        elif self.state.lower() == "test":
            data_directory = f"{mod}_registered_test_selection"

        elif self.state.lower() == "empty":
            data_directory = "nonexisting_dir_4t6yh"
        else:
            raise Exception("Unknown data state. Choose either 'train or 'test'")

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
        modality, patient_id = re.findall(r'(ct|mr)_train_(\d{4})_.*?.nii.gz', trailing_name)[0]
        patient_id = int(patient_id)

        # Generate cmrxmotion id like 001-02-ES
        mmwhs_id = f"{patient_id:04d}-{modality}"

        if not IMAGE_ID in trailing_name:
            label_paths[mmwhs_id] = str(_path)
        else:
            img_paths[mmwhs_id] = str(_path)

    if self.ensure_labeled_pairs:
        pair_idxs = set(img_paths).intersection(set(label_paths))
        label_paths = {_id: _path for _id, _path in label_paths.items() if _id in pair_idxs}
        img_paths = {_id: _path for _id, _path in img_paths.items() if _id in pair_idxs}

    img_data_3d = {}
    label_data_3d = {}
    modified_label_data_3d = {}
    additional_data_3d = {}

    # Load data from files
    print(f"Loading MMWHS {self.state} images and labels... ({modalities})")
    id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

    description = f"{len(img_paths)} images, {len(label_paths)} labels"

    for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
        trailing_name = str(_file).split("/")[-1]
        is_label = not IMAGE_ID in trailing_name
        nib_tmp = nib.load(_file)
        tmp = torch.from_numpy(nib_tmp.get_fdata()).squeeze()

        align_affine_path = str(Path(self.base_dir, "preprocessed", f"f1002mr_m{_3d_id.split('-')[0]}{_3d_id.split('-')[1]}.mat"))
        align_affine = torch.from_numpy(np.loadtxt(align_affine_path))
        affine = torch.as_tensor(nib_tmp.affine)

        additional_data_3d[_3d_id] = dict(nifti_affine=affine)

        if is_label:
            resample_mode = 'nearest'
            tmp = replace_label_values(tmp)
        else:
            resample_mode = 'trilinear'

        if self.do_resample:
            tmp = F.interpolate(tmp.unsqueeze(0).unsqueeze(0), size=self.resample_size, mode=resample_mode).squeeze(0).squeeze(0)

            if tmp.shape != self.resample_size:
                difs = np.array(self.resample_size) - torch.tensor(tmp.shape)
                pad_before, pad_after = (difs/2).clamp(min=0).int(), (difs.int()-(difs/2).int()).clamp(min=0)
                tmp = F.pad(tmp, tuple(torch.stack([pad_before.flip(0), pad_after.flip(0)], dim=1).view(-1).tolist()))

        if self.crop_3d_region is not None:
            difs = self.crop_3d_region[:,1] - torch.tensor(tmp.shape)
            pad_before, pad_after = (difs/2).clamp(min=0).int(), (difs.int()-(difs/2).int()).clamp(min=0)
            tmp = F.pad(tmp, tuple(torch.stack([pad_before.flip(0), pad_after.flip(0)], dim=1).view(-1).tolist()))

            tmp = tmp[self.crop_3d_region[0,0]:self.crop_3d_region[0,1], :, :]
            tmp = tmp[:, self.crop_3d_region[1,0]:self.crop_3d_region[1,1], :]
            tmp = tmp[:, :, self.crop_3d_region[2,0]:self.crop_3d_region[2,1]]

        if not IMAGE_ID in trailing_name:
            label_data_3d[_3d_id] = tmp.long()

        else:
            if self.do_normalize: # Normalize image to zero mean and unit std
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
    new_values =  [0,  1,   2,   0,   3,   4,   5,   0,   0  ]

    modified_label = label.clone()
    for orig, new in zip(orig_values, new_values):
        modified_label[modified_label == orig] = new
    return modified_label