import os
import json
import re
import torch

import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from joblib import Memory
import einops as eo

import monai

from slice_inflate.utils.common_utils import DotDict, get_script_dir
from slice_inflate.utils.torch_utils import ensure_dense
from slice_inflate.datasets.hybrid_id_dataset import HybridIdDataset
from slice_inflate.utils.nifti_utils import nifti_grid_sample, get_zooms
from slice_inflate.datasets.clinical_cardiac_views import get_clinical_cardiac_view_affines
from slice_inflate.utils.register_centroids import get_centroid_reorient_grid_affine
from slice_inflate.utils.nnunetv2_utils import get_segment_fn
from slice_inflate.utils.torch_utils import get_batch_score_per_label
from slice_inflate.utils.log_utils import log_oa_metrics, log_label_metrics

import monai

from slice_inflate.utils.common_utils import DotDict
from slice_inflate.utils.torch_utils import ensure_dense, get_batch_score_per_label, reduce_label_scores_epoch

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

        if kwargs['use_binarized_labels']:
            label_tags=("background", "foreground")

        self.nnunet_segment_model_path = "/home/weihsbach/storage/staff/christianweihsbach/nnunet/nnUNetV2_results/Dataset671_MMWHS_ac_focus/nnUNetTrainer_GIN_MultiRes__nnUNetPlans__2d"
        kwargs['nnunet_segment_model_path'] = self.nnunet_segment_model_path

        super().__init__(*args, state=state, label_tags=label_tags, **kwargs)

    def extract_3d_id(self, _input):
        return _input

    def extract_short_3d_id(self, _input):
        return _input

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

        if self.use_modified:
            modified_label = self.modified_label_data_3d.get(
                _id, label.detach().clone())
        else:
            modified_label = label.detach().clone()

        image = image.to(device=self.device)
        label = label.to(device=self.device)

        modified_label, _ = ensure_dense(modified_label)
        modified_label = modified_label.to(device=self.device)

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

        is_label = ('label' in type_str)
        return mmwhs_id, is_label


    def set_segment_fn(self, fold_idx):
        self.segment_fn = get_segment_fn(self.nnunet_segment_model_path, fold_idx, torch.device('cuda'))


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

        label_scores_dataset = {}
        fixed_ref_path = Path(THIS_SCRIPT_DIR, 'slice_inflate/datasets/ref_heart.nii.gz')

        for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
            additional_data_3d[_3d_id] = additional_data_3d.get(_3d_id, {})

            file_id, is_label = MMWHSDataset.get_file_id(_file)
            nib_tmp = nib.load(_file)
            tmp = torch.from_numpy(nib_tmp.get_fdata()).squeeze()
            if not is_label:
                tmp = tmp.float()
            loaded_nii_affine = torch.as_tensor(nib_tmp.affine)

            tmp, _, hires_nii_affine = nifti_grid_sample(
                tmp.unsqueeze(0).unsqueeze(0),
                loaded_nii_affine.view(1,4,4), ras_transform_mat=None,
                fov_mm=torch.as_tensor(self.hires_fov_mm), fov_vox=torch.as_tensor(self.hires_fov_vox),
                is_label=is_label,
                pre_grid_sample_affine=None,
                # pre_grid_sample_hidden_affine=None,
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
                    hires_nii_affine.view(1,4,4), ras_transform_mat=None,
                    fov_mm=torch.as_tensor(self.prescan_fov_mm), fov_vox=torch.as_tensor(self.prescan_fov_vox),
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

            if not is_label and self.clinical_view_affine_type == 'from-segmented':
                # Segment from image
                prescan_image, _, prescan_nii_affine = nifti_grid_sample(
                    tmp.unsqueeze(0).unsqueeze(0),
                    hires_nii_affine.view(1,4,4), ras_transform_mat=None,
                    fov_mm=torch.as_tensor(self.prescan_fov_mm), fov_vox=torch.as_tensor(self.prescan_fov_vox),
                    is_label=False,
                    pre_grid_sample_affine=None,
                    # pre_grid_sample_hidden_affine=None,
                    dtype=torch.float32
                )

                # Segment using nnunet v2 model
                lores_spacing = get_zooms(prescan_nii_affine)
                prescan_segmentation = segment_fn(prescan_image.cuda(), lores_spacing.view(1,3)).cpu()
                prescan_segmentation = prescan_segmentation.permute(0,3,1,2)# NNUNET does sth weird with the dimensions
                # permute(0,1,3,2).permute(0,2,1,3)
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

            elif is_label and self.clinical_view_affine_type == 'from-gt':
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

        # Initialize 3d modified labels as unmodified labels
        for label_id in label_data_3d.keys():
            modified_label_data_3d[label_id] = label_data_3d[label_id]

        return dict(
            img_paths=img_paths,
            label_paths=label_paths,
            img_data_3d=img_data_3d,
            label_data_3d=label_data_3d,
            modified_label_data_3d=modified_label_data_3d,
            additional_data_3d=additional_data_3d
        )
