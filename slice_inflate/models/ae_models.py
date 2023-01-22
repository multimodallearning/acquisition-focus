import torch

class BlendowskiAE(torch.nn.Module):

    class ConvBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels_list: list,
            strides_list: list, kernels_list:list=None, paddings_list:list=None, groups_list:list=None, mode='3d'):
            super().__init__()

            if mode == '3d':
                conv_op = torch.nn.Conv3d
                norm_op = torch.nn.BatchNorm3d
            elif mode == '2d':
                conv_op = torch.nn.Conv2d
                norm_op = torch.nn.BatchNorm2d
            else:
                raise ValueError()

            ops = []
            in_channels = [in_channels] + out_channels_list[:-1]
            if kernels_list is None:
                kernels_list = [3] * len(out_channels_list)
            if paddings_list is None:
                paddings_list = [1] * len(out_channels_list)
            if groups_list is None:
                groups_list = [1] * len(out_channels_list)

            for op_idx in range(len(out_channels_list)):
                ops.append(conv_op(
                    in_channels[op_idx],
                    out_channels_list[op_idx],
                    kernel_size=kernels_list[op_idx],
                    stride=strides_list[op_idx],
                    padding=paddings_list[op_idx],
                    groups=groups_list[op_idx]
                ))
                ops.append(norm_op(out_channels_list[op_idx]))
                ops.append(torch.nn.LeakyReLU())

            self.block = torch.nn.Sequential(*ops)

        def forward(self, x):
            return self.block(x)



    def __init__(self, in_channels, out_channels, decoder_in_channels=2, debug_mode=False):
        super().__init__()

        self.debug_mode = debug_mode

        self.first_layer_encoder = self.ConvBlock(in_channels, out_channels_list=[8], strides_list=[1])
        self.first_layer_decoder = self.ConvBlock(8, out_channels_list=[8,out_channels], strides_list=[1,1])

        self.second_layer_encoder = self.ConvBlock(8, out_channels_list=[20,20,20], strides_list=[2,1,1])
        self.second_layer_decoder = self.ConvBlock(20, out_channels_list=[8], strides_list=[1])

        self.third_layer_encoder = self.ConvBlock(20, out_channels_list=[40,40,40], strides_list=[2,1,1])
        self.third_layer_decoder = self.ConvBlock(40, out_channels_list=[20], strides_list=[1])

        self.fourth_layer_encoder = self.ConvBlock(40, out_channels_list=[60,60,60], strides_list=[2,1,1])
        self.fourth_layer_decoder = self.ConvBlock(decoder_in_channels, out_channels_list=[40], strides_list=[1])

        self.deepest_layer = torch.nn.Sequential(
            self.ConvBlock(60, out_channels_list=[60,40,20], strides_list=[2,1,1]),
            torch.nn.Conv3d(20, 2, kernel_size=1, stride=1, padding=0)
        )

        self.encoder = torch.nn.Sequential(
            self.first_layer_encoder,
            self.second_layer_encoder,
            self.third_layer_encoder,
            self.fourth_layer_encoder,
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            self.fourth_layer_decoder,
            torch.nn.Upsample(scale_factor=2),
            self.third_layer_decoder,
            torch.nn.Upsample(scale_factor=2),
            self.second_layer_decoder,
            torch.nn.Upsample(scale_factor=2),
            self.first_layer_decoder,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.deepest_layer(h)
        # h = debug_forward_pass(self.encoder, x, STEP_MODE=False)
        # h = debug_forward_pass(self.deepest_layer, h, STEP_MODE=False)
        return h

    def decode(self, z):
        if self.debug_mode:
            return debug_forward_pass(self.decoder, z, STEP_MODE=False)
        else:
            return self.decoder(z)

    def forward(self, x):
        # x = torch.nn.functional.instance_norm(x)
        z = self.encode(x)
        return self.decode(z), z



class BlendowskiVAE(BlendowskiAE):
    def __init__(self, std_max=10.0, epoch=0, epoch_reach_std_max=250, *args, **kwargs):
        kwargs['decoder_in_channels'] = 1
        super().__init__(*args, **kwargs)

        self.deepest_layer_upstream = self.ConvBlock(60, out_channels_list=[60,40,20], strides_list=[2,1,1])
        self.deepest_layer_downstream = nn.ModuleList([
            torch.nn.Conv3d(20, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv3d(20, 1, kernel_size=1, stride=1, padding=0)
        ])

        self.log_var_scale = nn.Parameter(torch.Tensor([0.0]))
        self.epoch = epoch
        self.epoch_reach_std_max = epoch_reach_std_max
        self.std_max = std_max

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_std_max(self):
        SIGMOID_XMIN, SIGMOID_XMAX = -8.0, 8.0
        s_x = (SIGMOID_XMAX-SIGMOID_XMIN) / (self.epoch_reach_std_max - 0) * self.epoch + SIGMOID_XMIN
        std_max = torch.sigmoid(torch.tensor(s_x)) * self.std_max
        return std_max

    def sample_z(self, mean, std):
        q = torch.distributions.Normal(mean, std)
        return q.rsample() # Caution, dont use torch.normal(mean=mean, std=std). Gradients are not backpropagated

    def encode(self, x):
        h = self.encoder(x)
        h = self.deepest_layer_upstream(h)
        mean = self.deepest_layer_downstream[0](h)
        log_var = self.deepest_layer_downstream[1](h)
        return mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)
        std = torch.exp(log_var/2)
        std = std.clamp(min=1e-10, max=self.get_std_max())
        z = self.sample_z(mean=mean, std=std)
        return self.decode(z), (z, mean, std)



class HybridAE(BlendowskiAE):
    def __init__(self, in_channels, out_channels, decoder_in_channels=2, debug_mode=False):
        super().__init__(in_channels, out_channels, decoder_in_channels, debug_mode)
        self.first_layer_encoder = self.ConvBlock(in_channels, out_channels_list=[16], kernels_list=[5,5,5], strides_list=[1], paddings_list=[2,2,2], groups_list=[2,2,2], mode='2d')
        self.second_layer_encoder = self.ConvBlock(16, out_channels_list=[40,40,40], kernels_list=[5,5,5], strides_list=[2,1,1], paddings_list=[2,2,2], groups_list=[2,2,2], mode='2d')
        self.third_layer_encoder = self.ConvBlock(40, out_channels_list=[80,80,80], kernels_list=[5,5,5], strides_list=[2,1,1], paddings_list=[2,2,2], groups_list=[2,2,2], mode='2d')
        self.fourth_layer_encoder = self.ConvBlock(80, out_channels_list=[120,120,120], kernels_list=[5,5,5], strides_list=[2,1,1], paddings_list=[2,2,2], groups_list=[2,2,2], mode='2d')

        self.pre_decoder_channels = 16

        self.encoder = torch.nn.Sequential(
            self.first_layer_encoder,
            self.second_layer_encoder,
            self.third_layer_encoder,
            self.fourth_layer_encoder,
        )

        self.deepest_layer = torch.nn.Sequential(
            self.ConvBlock(120, out_channels_list=[120,80,40], strides_list=[2,1,1], mode='2d'),
            torch.nn.Conv2d(40, self.pre_decoder_channels, kernel_size=1, stride=1, padding=0),
        )

        self.linear = torch.nn.Linear(self.pre_decoder_channels*8**2, 2*8**3)

    def forward(self, x):
        bsz = x.shape[0]
        h = self.encoder(x)
        h = self.deepest_layer(h)
        h = self.linear(h.view(bsz,-1))
        return self.decoder(h.view(-1, 2, 8, 8, 8)), h