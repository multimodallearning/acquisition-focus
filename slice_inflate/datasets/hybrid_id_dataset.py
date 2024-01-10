from abc import abstractmethod
import warnings
from collections.abc import Iterable
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

from slice_inflate.utils.torch_utils import interpolate_sample, augment_noise, spatial_augment, torch_manual_seeded, ensure_dense, get_bincounts
from slice_inflate.utils.common_utils import LabelDisturbanceMode

class HybridIdDataset(Dataset):

    def __init__(self,
        data_base_dir,
        ensure_labeled_pairs=True, do_resample=True,
        resample_size: tuple=(96,96,60), do_normalize:bool=True,
        max_load_3d_num=None,
        crop_3d_region=None, crop_around_3d_label_center=None,
        modified_3d_label_override=None,
        crop_2d_slices_gt_num_threshold=0,
        prevent_disturbance=False,
        label_tags=(),
        use_2d_normal_to=None, crop_around_2d_label_center=None,
        pre_interpolation_factor=2., device='cpu', debug=False,
        **kwargs
    ):

        # Prepare an attribute dict to identify all dataset settings for joblib
        self.self_attributes = locals().copy()
        for arg_name, arg_value in kwargs.items():
            self.self_attributes[arg_name] = arg_value

        del self.self_attributes['kwargs']
        del self.self_attributes['self']

        self.data_base_dir = data_base_dir
        self.ensure_labeled_pairs = ensure_labeled_pairs
        self.do_resample = do_resample
        self.resample_size = resample_size
        self.do_normalize = do_normalize
        self.max_load_3d_num = max_load_3d_num
        self.crop_3d_region = crop_3d_region
        self.use_2d_normal_to = use_2d_normal_to
        self.crop_2d_slices_gt_num_threshold = crop_2d_slices_gt_num_threshold
        self.prevent_disturbance = prevent_disturbance
        self.pre_interpolation_factor = pre_interpolation_factor
        self.device = device
        self.debug = debug

        self.label_tags = label_tags
        self.disturbed_idxs = []
        self.do_augment = False
        self.use_modified = False
        self.augment_at_collate = False

        # Load base 3D data
        all_3d_data_dict = self.load_data(self.self_attributes)

        self.self_attributes['img_paths'] = self.img_paths = all_3d_data_dict.pop('img_paths', {})
        self.self_attributes['label_paths'] = self.label_paths = all_3d_data_dict.pop('label_paths', {})
        self.self_attributes['img_data_3d'] = self.img_data_3d = all_3d_data_dict.pop('img_data_3d', {})
        self.self_attributes['label_data_3d'] = self.label_data_3d = all_3d_data_dict.pop('label_data_3d', {})
        self.self_attributes['modified_label_data_3d'] = self.modified_label_data_3d = all_3d_data_dict.pop('modified_label_data_3d', {})
        self.self_attributes['additional_data_3d'] = self.additional_data_3d = all_3d_data_dict.pop('additional_data_3d', {})

        # Postprocessing of 3d volumes
        print("Postprocessing 3D volumes")
        orig_3d_num = len(self.label_data_3d.keys())

        if self.ensure_labeled_pairs:
            labeled_keys = set(self.label_data_3d.keys())
            unlabelled_imgs = set(self.img_data_3d.keys()) - labeled_keys
            unlabelled_modified_labels = set([self.extract_3d_id(key) for key in self.modified_label_data_3d.keys()]) - labeled_keys

            for del_key in unlabelled_imgs:
                del self.img_data_3d[del_key]
            for del_key in unlabelled_modified_labels:
                del self.modified_label_data_3d[del_key]

        if self.max_load_3d_num:
            for del_key in sorted(list(self.img_data_3d.keys()))[self.max_load_3d_num:]:
                del self.img_data_3d[del_key]
            for del_key in sorted(list(self.label_data_3d.keys()))[self.max_load_3d_num:]:
                del self.label_data_3d[del_key]
            for del_key in sorted(list(self.modified_label_data_3d.keys()))[self.max_load_3d_num:]:
                del self.modified_label_data_3d[del_key]

        postprocessed_3d_num = len(self.label_data_3d.keys())

        print(f"Removed {orig_3d_num - postprocessed_3d_num} 3D images in postprocessing")
        #check for consistency
        print(f"Equal image and label numbers: {set(self.img_data_3d)==set(self.label_data_3d)==set(self.modified_label_data_3d)} ({len(self.img_data_3d)})")

        self.bincounts_3d = {}

        for _3d_id, label  in self.label_data_3d.items():
            self.bincounts_3d[_3d_id] = get_bincounts(label, len(self.label_tags))

        # Retrieve slices and plugin modified data
        self.img_data_2d = {}
        self.label_data_2d = {}
        self.modified_label_data_2d = {}

        if use_2d_normal_to is not None:
            if extract_slice_func is None:
                extract_slice_func = HybridIdDataset.extract_2d_data

            all_2d_data_dict = extract_slice_func(self.self_attributes)
            self.img_data_2d = all_2d_data_dict.pop('img_data_2d', {})
            self.label_data_2d = all_2d_data_dict.pop('label_data_2d', {})
            self.modified_label_data_2d = all_2d_data_dict.pop('modified_label_data_2d', {})

        # Now make sure dicts are ordered
        self.img_paths = OrderedDict(sorted(self.img_paths.items()))
        self.label_paths = OrderedDict(sorted(self.label_paths.items()))
        self.img_data_3d = OrderedDict(sorted(self.img_data_3d.items()))
        self.label_data_3d = OrderedDict(sorted(self.label_data_3d.items()))
        self.modified_label_data_3d = OrderedDict(sorted(self.modified_label_data_3d.items()))
        self.img_data_2d = OrderedDict(sorted(self.img_data_2d.items()))
        self.label_data_2d = OrderedDict(sorted(self.label_data_2d.items()))
        self.modified_label_data_2d = OrderedDict(sorted(self.modified_label_data_2d.items()))

        print("Data import finished.")
        print(f"Dataloader will yield {'2D' if self.use_2d_normal_to else '3D'} samples")

    def extract_3d_id(self, _input):
        raise NotImplementedError()

    def extract_short_3d_id(self, _input):
        raise NotImplementedError()

    def get_short_3d_ids(self):
        return [self.extract_short_3d_id(_id) for _id in self.get_3d_ids()]

    def get_3d_ids(self):
        return list(self.img_data_3d.keys())

    def get_2d_ids(self):
        assert self.use_2d(), "Dataloader does not provide 2D data."
        return list(self.img_data_2d.keys())

    def get_id_dicts(self, use_2d_override=None):
        all_3d_ids = self.get_3d_ids()
        id_dicts = []
        if self.use_2d(use_2d_override):
            for _2d_dataset_idx, _2d_id in enumerate(self.get_2d_ids()):
                _3d_id = self.get_3d_from_2d_identifiers(_2d_id)
                id_dicts.append(
                    {
                        '2d_id': _2d_id,
                        '2d_dataset_idx': _2d_dataset_idx,
                        '3d_id': _3d_id,
                        '3d_dataset_idx': all_3d_ids.index(_3d_id),
                    }
                )
        else:
            for _3d_dataset_idx, _3d_id in enumerate(self.get_3d_ids()):
                id_dicts.append(
                    {
                        '3d_id': _3d_id,
                        '3d_dataset_idx': all_3d_ids.index(_3d_id),
                    }
                )

        return id_dicts

    def switch_2d_identifiers(self, _2d_identifiers):
        assert self.use_2d(), "Dataloader does not provide 2D data."

        if isinstance(_2d_identifiers, (torch.Tensor, np.ndarray)):
            _2d_identifiers = _2d_identifiers.tolist()
        elif not isinstance(_2d_identifiers, Iterable) or isinstance(_2d_identifiers, str):
            _2d_identifiers = [_2d_identifiers]

        _ids = self.get_2d_ids()
        if all([isinstance(elem, int) for elem in _2d_identifiers]):
            vals = [_ids[elem] for elem in _2d_identifiers]
        elif all([isinstance(elem, str) for elem in _2d_identifiers]):
            vals = [_ids.index(elem) for elem in _2d_identifiers]
        else:
            raise ValueError
        return vals[0] if len(vals) == 1 else vals

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

    def get_3d_from_2d_identifiers(self, _2d_identifiers, retrn='id'):
        assert self.use_2d(), "Dataloader does not provide 2D data."
        assert retrn in ['id', 'idx']

        if isinstance(_2d_identifiers, (torch.Tensor, np.ndarray)):
            _2d_identifiers = _2d_identifiers.tolist()
        elif not isinstance(_2d_identifiers, Iterable) or isinstance(_2d_identifiers, str):
            _2d_identifiers = [_2d_identifiers]

        if isinstance(_2d_identifiers[0], int):
            _2d_identifiers = self.switch_2d_identifiers(_2d_identifiers)

        vals = []
        for item in _2d_identifiers:
            _3d_id = self.extract_3d_id(item)
            if retrn == 'id':
                vals.append(_3d_id)
            elif retrn == 'idx':
                vals.append(self.switch_3d_identifiers(_3d_id))

        return vals[0] if len(vals) == 1 else vals

    def use_2d(self, override=None):
        if not self.use_2d_normal_to:
            return False
        elif override is not None:
            return override
        else:
            return True

    def __len__(self, use_2d_override=None):
        if self.use_2d(use_2d_override):
            return len(self.img_data_2d)
        return len(self.img_data_3d)

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

        spat_augment_grid = []

        if self.use_modified:
            if use_2d:
                modified_label = self.modified_label_data_2d.get(_id, label.detach().clone())
            else:
                modified_label = self.modified_label_data_3d.get(_id, label.detach().clone())
        else:
            modified_label = label.detach().clone()

        b_image = image.unsqueeze(0).to(device=self.device)
        b_label = label.unsqueeze(0).to(device=self.device)
        modified_label, _ = ensure_dense(modified_label)
        b_modified_label = modified_label.unsqueeze(0).to(device=self.device)

        if self.do_augment and not self.augment_at_collate:
            b_image, b_label, b_spat_augment_grid = self.augment(
                b_image, b_label, use_2d, pre_interpolation_factor=self.pre_interpolation_factor
            )
            _, b_modified_label, _ = spatial_augment(
                b_label=b_modified_label, use_2d=use_2d, b_grid_override=b_spat_augment_grid,
                pre_interpolation_factor=self.pre_interpolation_factor
            )
            spat_augment_grid = b_spat_augment_grid.squeeze(0).detach().cpu().clone()

        elif not self.do_augment:
            b_image, b_label = interpolate_sample(b_image, b_label, self.pre_interpolation_factor, use_2d)
            _, b_modified_label = interpolate_sample(b_label=b_modified_label, scale_factor=self.pre_interpolation_factor,
                use_2d=use_2d)

        image = b_image.squeeze(0).cpu()
        label = b_label.squeeze(0).cpu()
        modified_label = b_modified_label.squeeze(0).cpu()

        if label.numel() == 0:
            label = torch.zeros_like(image)
        if modified_label.numel() == 0:
            modified_label = torch.zeros_like(image)

        if use_2d:
            assert image.dim() == label.dim() == 2
        else:
            assert image.dim() == label.dim() == 3

        return {
            'image': image,
            'label': label,
            'modified_label': modified_label,
            # if disturbance is off, modified label is equals label
            'dataset_idx': dataset_idx,
            'id': _id,
            'image_path': image_path,
            'label_path': label_path,
            'spat_augment_grid': spat_augment_grid,
            'additional_data': additional_data
        }

    def get_3d_item(self, _3d_dataset_id):
        return self.__getitem__(_3d_dataset_id, use_2d_override=False)

    def get_data(self, use_2d_override=None):
        if self.use_2d(use_2d_override):
            img_stack = torch.stack(list(self.img_data_2d.values()), dim=0)
            label_stack = torch.stack(list(self.label_data_2d.values()), dim=0)
            modified_label_stack = torch.stack(list(self.modified_label_data_2d.values()), dim=0)
        else:
            img_stack = torch.stack(list(self.img_data_3d.values()), dim=0)
            label_stack = torch.stack(list(self.label_data_3d.values()), dim=0)
            modified_label_stack = torch.stack(list(self.modified_label_data_3d.values()), dim=0)

        return img_stack, label_stack, modified_label_stack

    def disturb_idxs(self, all_idxs, disturbance_mode, disturbance_strength=1., use_2d_override=None):
        if self.prevent_disturbance:
            warnings.warn("Disturbed idxs shall be set but disturbance is prevented for dataset.")
            return

        use_2d = self.use_2d(use_2d_override)

        if all_idxs is not None:
            if isinstance(all_idxs, (np.ndarray, torch.Tensor)):
                all_idxs = all_idxs.tolist()

            self.disturbed_idxs = all_idxs
        else:
            self.disturbed_idxs = []

        # Reset modified data
        for idx in range(self.__len__(use_2d_override=use_2d)):
            if use_2d:
                label_id = self.get_2d_ids()[idx]
                self.modified_label_data_2d[label_id] = self.label_data_2d[label_id]
            else:
                label_id = self.get_3d_ids()[idx]
                self.modified_label_data_3d[label_id] = self.label_data_3d[label_id]

            # Now apply disturbance
            if idx in self.disturbed_idxs:
                if use_2d:
                    label = self.modified_label_data_2d[label_id].detach().clone()
                else:
                    label = self.modified_label_data_3d[label_id].detach().clone()

                with torch_manual_seeded(idx):
                    if str(disturbance_mode)==str(LabelDisturbanceMode.FLIP_ROLL):
                        roll_strength = 10*disturbance_strength
                        if use_2d:
                            modified_label = \
                                torch.roll(
                                    label.transpose(-2,-1),
                                    (
                                        int(torch.randn(1)*roll_strength),
                                        int(torch.randn(1)*roll_strength)
                                    ),(-2,-1)
                                )
                        else:
                            modified_label = \
                                torch.roll(
                                    label.permute(1,2,0),
                                    (
                                        int(torch.randn(1)*roll_strength),
                                        int(torch.randn(1)*roll_strength),
                                        int(torch.randn(1)*roll_strength)
                                    ),(-3,-2,-1)
                                )

                    elif str(disturbance_mode)==str(LabelDisturbanceMode.AFFINE):
                        b_modified_label = label.unsqueeze(0).to(device=self.device)
                        _, b_modified_label, _ = spatial_augment(b_label=b_modified_label, use_2d=use_2d,
                            bspline_num_ctl_points=6, bspline_strength=0., bspline_probability=0.,
                            affine_strength=0.09*disturbance_strength,
                            add_affine_translation=0.18*disturbance_strength, affine_probability=1.)
                        modified_label = b_modified_label.squeeze(0).cpu()

                    else:
                        raise ValueError(f"Disturbance mode {disturbance_mode} is not implemented.")

                    if use_2d:
                        self.modified_label_data_2d[label_id] = modified_label
                    else:
                        self.modified_label_data_3d[label_id] = modified_label


    def train(self, augment=True, use_modified=True):
        self.do_augment = augment
        self.use_modified = use_modified

    def eval(self, augment=False, use_modified=False):
        self.train(augment, use_modified)

    def set_augment_at_collate(self, augment_at_collate=True):
        self.augment_at_collate = augment_at_collate

    def get_efficient_augmentation_collate_fn(self):
        use_2d = True if self.use_2d_normal_to else False

        def collate_closure(batch):
            batch = torch.utils.data._utils.collate.default_collate(batch)
            if self.augment_at_collate and self.do_augment:
                # Augment the whole batch not just one sample
                b_image = batch['image'].to(device=self.device)
                b_label = batch['label'].to(device=self.device)
                b_modified_label = batch['modified_label'].to(device=self.device)

                b_image, b_label, b_spat_augment_grid = self.augment(
                    b_image, b_label, use_2d, pre_interpolation_factor=self.pre_interpolation_factor
                )
                _, b_modified_label, _ = spatial_augment(
                    b_label=b_modified_label, use_2d=use_2d, b_grid_override=b_spat_augment_grid,
                    pre_interpolation_factor=self.pre_interpolation_factor
                )
                b_spat_augment_grid = b_spat_augment_grid.detach().clone()
                batch['image'], batch['label'], batch['modified_label'], batch['spat_augment_grid'] = b_image, b_label, b_modified_label, b_spat_augment_grid

            return batch

        return collate_closure

    def augment(self, b_image, b_label, use_2d,
        noise_strength=0.05,
        bspline_num_ctl_points=6, bspline_strength=0.03, bspline_probability=.95,
        affine_strength=0.2, affine_probability=.45,
        pre_interpolation_factor=2.):

        if use_2d:
            assert b_image.dim() == b_label.dim() == 3, \
                f"Augmenting 2D. Input batch of image and " \
                f"label should be BxHxW but are {b_image.shape} and {b_label.shape}"
        else:
            assert b_image.dim() == b_label.dim() == 4, \
                f"Augmenting 3D. Input batch of image and " \
                f"label should be BxDxHxW but are {b_image.shape} and {b_label.shape}"

        b_image = augment_noise(b_image, strength=noise_strength)
        b_image, b_label, b_spat_augment_grid = spatial_augment(
            b_image, b_label,
            bspline_num_ctl_points=bspline_num_ctl_points, bspline_strength=bspline_strength, bspline_probability=bspline_probability,
            affine_strength=affine_strength, affine_probability=affine_probability,
            pre_interpolation_factor=pre_interpolation_factor, use_2d=use_2d)

        b_label = b_label.long()

        return b_image, b_label, b_spat_augment_grid

    @abstractmethod
    def get_file_id(file_path):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def extract_2d_data(self_attributes: dict):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_data(self_attributes: dict):
        raise NotImplementedError()