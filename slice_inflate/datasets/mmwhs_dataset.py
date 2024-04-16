import re
from pathlib import Path
import torch

from slice_inflate.datasets.base_dataset import BaseDataset
from slice_inflate.utils.nnunetv2_utils import get_segment_fn



class MMWHSDataset(BaseDataset):
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
        base_segment_fn = get_segment_fn(self.nnunet_segment_model_path, fold_idx, torch.device('cuda'))

        def segment_fn(input):
            return base_segment_fn(input).permute(0,3,1,2) # Additional permutation is needed for MMWHS (nnUNet related error)

        self.segment_fn = segment_fn