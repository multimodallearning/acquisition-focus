import os
import time
import glob
import re
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from joblib import Memory

from slice_inflate.utils.common_utils import DotDict
from slice_inflate.utils.torch_utils import ensure_dense, restore_sparsity
from slice_inflate.datasets.hybrid_id_dataset import HybridIdDataset

cache = Memory(location=os.environ['MMWHS_CACHE_PATH'])

class MMWHSDataset(HybridIdDataset):
    def __init__(self, *args, state='training',
        label_tags=(
            "background",
            "left_myocardium",
            "left_atrium",
            "left_ventricle",
            "right_atrium",
            "right_ventricle",
            "ascending_aorta",
            "pulmonary_artery"),
        **kwargs):
        self.state = state

        super().__init__(*args, state=state, label_tags=label_tags, **kwargs)

    def extract_3d_id(self, _input):
        # Match sth like 1001-HLA:mTS1
        items = re.findall(r'^(\d{4})-[(CT)(MRI)](:m[A-Z0-9a-z]{3,4})?', _input)[0]
        items = list(filter(None, items))
        return "-".join(items)

    def extract_short_3d_id(self, _input):
        # Match sth like 1001-HLA:mTS1 and returns 1001-HLA
        items = re.findall(r'^(\d{4})-(HLA)', _input)[0]
        items = list(filter(None, items))
        return "-".join(items)

# @cache.cache(verbose=True)
def load_data(self_attributes: dict):
    # Use only specific attributes of a dict to have a cacheable function
    self = DotDict(self_attributes)

    t0 = time.time()

    if self.modality == 'all':
        modality = ['mr', 'ct']
    else:
        modalities= list(self.modality)

    files = []

    for mod in modalities:
        if self.state.lower() == "training":
            data_directory = f"data/{mod}_train"

        elif self.state.lower() == "test":
            data_directory = f"data/{mod}_test"

        else:
            raise Exception("Unknown data state. Choose either 'training or 'test'")

        data_path = Path(self.base_dir, data_directory)

        if self.crop_3d_region is not None:
            self.crop_3d_region = torch.from_numpy(self.crop_3d_region)

        files.append(list(data_path.glob("**/*.nii.gz")))

    files = sorted(files)

    # First read filepaths
    img_paths = {}
    label_paths = {}

    if self.debug:
        files = files[:10]

    for _path in files:
        trailing_name = str(_path).split("/")[-1]
        # Extract ids from sth. like P001-1-ED-label.nii.gz
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

    # Load data from files
    print(f"Loading MMWHS {self.state} images and labels... ({modalities})")
    id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

    description = f"{len(img_paths)} images, {len(label_paths)} labels"

    for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
        trailing_name = str(_file).split("/")[-1]
        tmp = torch.from_numpy(nib.load(_file).get_fdata()).squeeze()

        if not IMAGE_ID in trailing_name:
            resample_mode = 'nearest'
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
    # None

    return dict(
        img_paths=img_paths,
        label_paths=label_paths,
        img_data_3d=img_data_3d,
        label_data_3d=label_data_3d,
        modified_label_data_3d=modified_label_data_3d,
    )

    def extract_2d_data(self):
        if use_2d_normal_to == "D":
            slice_dim = -3
        elif use_2d_normal_to == "H":
            slice_dim = -2
        elif use_2d_normal_to == "W":
            slice_dim = -1
        elif use_2d_normal_to == "HLA/SA":
            pass
        else:
            raise ValueError
        if use_2d_normal_to == "HLA/SA":
            for _3d_id, image in self.img_data_3d.items():
                raise NotImplementedError

        else:
            for _3d_id, image in self.img_data_3d.items():
                for idx, img_slc in [(slice_idx, image.select(slice_dim, slice_idx)) \
                                        for slice_idx in range(image.shape[slice_dim])]:
                    # Set data view for id like "003rW100"
                    self.img_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = img_slc

            for _3d_id, label in self.label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                        for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for id like "003rW100"
                    self.label_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = lbl_slc

            for _3d_id, label in self.modified_label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                        for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for id like "003rW100"
                    self.modified_label_data_2d[f"{_3d_id}:{use_2d_normal_to}{idx:03d}"] = lbl_slc


        # Postprocessing of 2d slices
        print("Postprocessing 2D slices")
        orig_2d_num = len(self.img_data_2d.keys())

        if self.crop_2d_slices_gt_num_threshold > 0:
            for key, label in list(self.label_data_2d.items()):
                uniq_vals = label.unique()

                if sum(label[label > 0]) < self.crop_2d_slices_gt_num_threshold:
                    # Delete 2D slices with less than n gt-pixels (but keep 3d data)
                    del self.img_data_2d[key]
                    del self.label_data_2d[key]
                    del self.modified_label_data_2d[key]

        postprocessed_2d_num = len(self.img_data_2d.keys())
        print(f"Removed {orig_2d_num - postprocessed_2d_num} of {orig_2d_num} 2D slices in postprocessing")

        nonzero_lbl_percentage = torch.tensor([lbl.sum((-2,-1)) > 0 for lbl in self.label_data_2d.values()]).sum()
        nonzero_lbl_percentage = nonzero_lbl_percentage/len(self.label_data_2d)
        print(f"Nonzero 2D labels: " f"{nonzero_lbl_percentage*100:.2f}%")

        nonzero_mod_lbl_percentage = torch.tensor([ensure_dense(lbl)[0].sum((-2,-1)) > 0 for lbl in self.modified_label_data_2d.values()]).sum()
        nonzero_mod_lbl_percentage = nonzero_mod_lbl_percentage/len(self.modified_label_data_2d)
        print(f"Nonzero modified 2D labels: " f"{nonzero_mod_lbl_percentage*100:.2f}%")
        print(f"Loader will use {postprocessed_2d_num} of {orig_2d_num} 2D slices.")


def cut_sa_hla_slice_from_volume(volume, _id):
    slc = None
    return slc