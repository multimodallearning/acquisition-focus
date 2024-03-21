from abc import abstractmethod
from collections.abc import Iterable
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

from slice_inflate.utils.nnunetv2_utils import get_segment_fn
from pathlib import Path
import json

class HybridIdDataset(Dataset):

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
        self.use_modified = False

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

        postprocessed_3d_num = len(self.label_data_3d.keys())

        print(f"Removed {orig_3d_num - postprocessed_3d_num} 3D images in postprocessing")
        #check for consistency
        print(f"Equal image and label numbers: {set(self.img_data_3d)==set(self.label_data_3d)==set(self.modified_label_data_3d)} ({len(self.img_data_3d)})")

        # Now make sure dicts are ordered
        self.img_paths = OrderedDict(sorted(self.img_paths.items()))
        self.label_paths = OrderedDict(sorted(self.label_paths.items()))
        self.img_data_3d = OrderedDict(sorted(self.img_data_3d.items()))
        self.label_data_3d = OrderedDict(sorted(self.label_data_3d.items()))
        self.modified_label_data_3d = OrderedDict(sorted(self.modified_label_data_3d.items()))

        print("Data import finished.")

    def extract_3d_id(self, _input):
        raise NotImplementedError()

    def extract_short_3d_id(self, _input):
        raise NotImplementedError()

    def get_short_3d_ids(self):
        return [self.extract_short_3d_id(_id) for _id in self.get_3d_ids()]

    def get_3d_ids(self):
        return list(self.img_data_3d.keys())

    def get_id_dicts(self):
        all_3d_ids = self.get_3d_ids()
        id_dicts = []

        for _3d_dataset_idx, _3d_id in enumerate(self.get_3d_ids()):
            id_dicts.append(
                {
                    '3d_id': _3d_id,
                    '3d_dataset_idx': all_3d_ids.index(_3d_id),
                }
            )

        return id_dicts

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

    @abstractmethod
    def __getitem__(self, dataset_id):
        raise NotImplementedError()

    def get_3d_item(self, _3d_dataset_id):
        return self.__getitem__(_3d_dataset_id)

    def train(self, use_modified=True):
        self.use_modified = use_modified

    def eval(self, augment=False, use_modified=False):
        self.train(augment, use_modified)

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