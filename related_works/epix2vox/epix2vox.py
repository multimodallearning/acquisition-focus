# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models
from torch.optim import Adam

from slice_inflate.utils.common_utils import DotDict

from related_works.epix2vox.models.encoder_128 import Encoder as Encoder128
from related_works.epix2vox.models.decoder_128 import Decoder as Decoder128
from related_works.epix2vox.models.merger_128 import Merger as Merger128
from related_works.epix2vox.models.refiner_128 import Refiner as Refiner128


from related_works.epix2vox.models.encoder_64 import Encoder as Encoder64
from related_works.epix2vox.models.decoder_64 import Decoder as Decoder64
from related_works.epix2vox.models.merger_64 import Merger as Merger64
from related_works.epix2vox.models.refiner_64 import Refiner as Refiner64


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def get_optimizer_and_scheduler(encoder, decoder, merger, refiner):
    cfg = DotDict(
        TRAIN=DotDict(
            ENCODER_LEARNING_RATE = 1e-3,
            DECODER_LEARNING_RATE = 1e-3,
            REFINER_LEARNING_RATE = 1e-3,
            MERGER_LEARNING_RATE = 1e-4,
            LR_MILESTONES = [150],
            BETAS = (.9, .999),
            GAMMA = .5,
        )
    )
    param_groups = [
        dict(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS),
        dict(params=decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS),
        dict(params=refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS),
        dict(params=merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE , betas=cfg.TRAIN.BETAS)
    ]

    solver = Adam(param_groups, lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver,
        milestones=cfg.TRAIN.LR_MILESTONES,
        gamma=cfg.TRAIN.GAMMA
    )

    return solver, lr_scheduler


class EPix2VoxModel128(torch.nn.Module):
    def __init__(self,
        use_merger=True, use_refiner=True, n_views=9,
        use_epix2vox=True,
        epoch_start_use_merger=0, epoch_start_use_refiner=0):

        super().__init__()
        cfg = DotDict(
            CONST=DotDict(N_VIEWS_RENDERING=n_views),
            NETWORK=DotDict(
                LEAKY_VALUE=.2,
                TCONV_USE_BIAS=False,
                USE_EP2V=use_epix2vox,
            )
        )

        self.encoder = Encoder128(cfg)
        self.decoder = Decoder128(cfg)
        self.refiner = Refiner128(cfg)
        self.merger = Merger128(cfg)

        self.use_merger = use_merger
        self.use_refiner = use_refiner

        self.epoch_start_use_merger = epoch_start_use_merger
        self.epoch_start_use_refiner = epoch_start_use_refiner

    def forward(self, views, epoch_idx):
        B,N_VIEWS,N_CHAN,SPAT_H,SPAT_W = views.shape
        assert SPAT_H == SPAT_W == 224
        assert N_CHAN == 3
        image_features = self.encoder(views)
        raw_features, generated_volume = self.decoder(image_features)

        if self.use_merger and epoch_idx >= self.epoch_start_use_merger:
            generated_volume = self.merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)

        return generated_volume



class EPix2VoxModel64(torch.nn.Module):
    def __init__(self,
        use_merger=True, use_refiner=True, n_views=9,
        epoch_start_use_merger=0, epoch_start_use_refiner=0):

        super().__init__()
        cfg = DotDict(
            CONST=DotDict(N_VIEWS_RENDERING=n_views),
            NETWORK=DotDict(
                LEAKY_VALUE=.2,
                TCONV_USE_BIAS=False,
                USE_EP2V=True)
        )

        self.encoder = Encoder64(cfg)
        self.decoder = Decoder64(cfg)
        self.refiner = Refiner64(cfg)
        self.merger = Merger64(cfg)

        self.use_merger = use_merger
        self.use_refiner = use_refiner

        self.epoch_start_use_merger = epoch_start_use_merger
        self.epoch_start_use_refiner = epoch_start_use_refiner

    def forward(self, views, epoch_idx):
        B,N_VIEWS,N_CHAN,SPAT_H,SPAT_W = views.shape
        assert SPAT_H == SPAT_W == 224
        assert N_CHAN == 3
        image_features = self.encoder(views)
        raw_features, generated_volume = self.decoder(image_features)

        if self.use_merger and epoch_idx >= self.epoch_start_use_merger:
            generated_volume = self.merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)

        return generated_volume