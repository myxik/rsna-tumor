import torch.nn as nn
import torch


class UNet3D(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        img_size: int = 512,
        base_n_filter: int = 8,
    ) -> None:
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode="nearest")
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(
            self.in_channels,
            self.base_n_filter,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv3d_c1_2 = nn.Conv3d(
            self.base_n_filter,
            self.base_n_filter,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(
            self.base_n_filter,
            self.base_n_filter * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(
            self.base_n_filter * 2, self.base_n_filter * 2
        )
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(
            self.base_n_filter * 2,
            self.base_n_filter * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(
            self.base_n_filter * 4, self.base_n_filter * 4
        )
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(
            self.base_n_filter * 4,
            self.base_n_filter * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(
            self.base_n_filter * 8, self.base_n_filter * 8
        )
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(
            self.base_n_filter * 8,
            self.base_n_filter * 16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(
            self.base_n_filter * 16, self.base_n_filter * 16
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = (
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 16, self.base_n_filter * 8
            )
        )

        self.conv3d_l0 = nn.Conv3d(
            self.base_n_filter * 8,
            self.base_n_filter * 8,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(
            self.base_n_filter * 16, self.base_n_filter * 16
        )
        self.conv3d_l1 = nn.Conv3d(
            self.base_n_filter * 16,
            self.base_n_filter * 8,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = (
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 8, self.base_n_filter * 4
            )
        )

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(
            self.base_n_filter * 8, self.base_n_filter * 8
        )
        self.conv3d_l2 = nn.Conv3d(
            self.base_n_filter * 8,
            self.base_n_filter * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = (
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 4, self.base_n_filter * 2
            )
        )

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(
            self.base_n_filter * 4, self.base_n_filter * 4
        )
        self.conv3d_l3 = nn.Conv3d(
            self.base_n_filter * 4,
            self.base_n_filter * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = (
            self.norm_lrelu_upscale_conv_norm_lrelu(
                self.base_n_filter * 2, self.base_n_filter
            )
        )

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(
            self.base_n_filter * 2, self.base_n_filter * 2
        )
        self.conv3d_l4 = nn.Conv3d(
            self.base_n_filter * 2,
            self.n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.ds2_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 8,
            self.n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.ds3_1x1_conv3d = nn.Conv3d(
            self.base_n_filter * 4,
            self.n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.conv3d_last = nn.Conv3d(
            self.base_n_filter * 8, 16, kernel_size=2, stride=4, padding=0, bias=False
        )  # CHANGE KERNEL_SIZE IN CASE OF NUM_SLICES CHANGE

        self.lin = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Linear(2048, 784),
            self.lrelu,
            nn.Linear(784, self.n_classes),
        )

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(
                feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU(),
        )

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(
                feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(
                feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # should be feat_in*2 or feat_in
            nn.Conv3d(
                feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        out = self.conv3d_last(out)
        out = out.view(out.shape[0], -1)
        out = self.lin(out)
        return out
