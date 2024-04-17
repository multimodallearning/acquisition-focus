import torch
from torch.utils.checkpoint import checkpoint
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from slice_inflate.utils.nifti_utils import get_zooms, rescale_rot_components_with_diag
from slice_inflate.utils.torch_utils import set_module


class HybridUnet(PlainConvUNet):
    def __init__(self, n_views, num_classes):

        n_stages = 6
        features_per_stage=[n_views*c for c in [16,32,64,128,256,256]]

        super().__init__(
            input_channels=n_views*num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=n_stages*[[3,3,3]],
            strides=[[1,1,1]]+(n_stages-1)*[[2,2,2]],
            n_conv_per_stage=n_stages*[2],
            num_classes=num_classes,
            n_conv_per_stage_decoder=(n_stages-1)*[2],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs=dict(eps=1e-05, affine=True),
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.modules.activation.LeakyReLU,
            nonlin_kwargs=dict(inplace=True),
            deep_supervision=False,
            nonlin_first=False
        )

        self.skip_connector = SkipConnector(n_views)
        self.__setup_2d_encoder__(n_views)
        self.n_views = n_views

    def forward(self, x, b_grid_affines):
        skips = self.encoder(x)
        embedded_skips = [self.skip_connector(s, b_grid_affines) for s in skips]
        return self.decoder(embedded_skips)

    def __setup_2d_encoder__(self, n_views):
        for keychain, mod in self.encoder.named_modules(remove_duplicate=False):
            if isinstance(mod, torch.nn.modules.conv.Conv3d):
                replacer = torch.nn.Conv2d(mod.in_channels, mod.out_channels,
                                           kernel_size=mod.kernel_size[:2],
                                           stride=mod.stride[:2], padding=mod.padding[:2],
                                           groups=n_views)

            elif isinstance(mod, torch.nn.modules.instancenorm.InstanceNorm3d):
                replacer = torch.nn.InstanceNorm2d(mod.num_features,
                                                   mod.eps, mod.momentum, mod.affine,
                                                   mod.track_running_stats)
            else:
                continue

            set_module(self.encoder, keychain, replacer)




class SkipConnector(torch.nn.Module):
    def __init__(self, n_views):
        self.dtype = torch.float32
        self.n_views = n_views
        super().__init__()

    def forward(self, x, b_grid_affines):
        B,C,SPAT,_ = x.shape
        C_PER_SLICE = C//self.n_views
        target_shape = torch.Size([B,C_PER_SLICE,SPAT,SPAT,SPAT])
        x_mid = torch.zeros(B,C,SPAT,SPAT,SPAT).to(x)
        x_mid[..., SPAT//2] = x # Embed slice in center of last dimension
        x_views = torch.chunk(x_mid, self.n_views, dim=1)

        reembed_views = []
        for vx, ga in zip(x_views,b_grid_affines):
            rescaled_affines = ga
            # Rescale to sample from volume slice space into volume space (forward grid sampling sampled to single slice, non-stacked)
            rescaled_affines = rescale_rot_components_with_diag(rescaled_affines, 1/get_zooms(rescaled_affines))

            reembed_grid = torch.nn.functional.affine_grid(
                rescaled_affines.to(self.dtype).inverse()[:,:3,:].view(B,3,4), target_shape, align_corners=False
            )
            reembed_vx = checkpoint(torch.nn.functional.grid_sample,
                vx, reembed_grid.to(vx), 'bilinear', 'zeros', False
            )
            reembed_views.append(reembed_vx)

        skip_out = torch.cat(reembed_views, dim=1)
        return skip_out