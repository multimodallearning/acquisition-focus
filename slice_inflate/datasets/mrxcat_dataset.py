import re
from pathlib import Path
import torch

from slice_inflate.datasets.base_dataset import BaseDataset
from slice_inflate.utils.nnunetv2_utils import get_segment_fn



class MRXCATDataset(BaseDataset):
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

        self.nnunet_segment_model_path = "/home/weihsbach/storage/staff/christianweihsbach/nnunet/nnUNetV2_results/Dataset670_MRXCAT_ac_focus/nnUNetTrainer_GIN_MultiRes__nnUNetPlans__2d"
        kwargs['nnunet_segment_model_path'] = self.nnunet_segment_model_path

        super().__init__(*args, state=state, label_tags=label_tags, **kwargs)

    def extract_3d_id(self, _input):
        return _input[:8]

    def get_file_id(file_path):
        file_path = Path(file_path)
        patient_id, temporary_frame_idx, type_str = re.findall(
            r'phantom_(\d{3})_t(\d{3})_(.*?).nii.gz', file_path.name)[0]
        patient_id = int(patient_id)
        temporary_frame_idx = int(temporary_frame_idx)
        mrxcat_id = f"{patient_id:03d}_t{temporary_frame_idx:03d}"

        is_label = type_str == 'label'
        return mrxcat_id, is_label

    def set_segment_fn(self, fold_idx):
        self.segment_fn = get_segment_fn(self.nnunet_segment_model_path, 0, torch.device('cuda'))