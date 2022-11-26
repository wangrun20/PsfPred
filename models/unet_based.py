import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from general_utils import rectangular_closure


class CA(nn.Module):
    """ChannelAttention, would not change (N, C, H, W)"""

    def __init__(self, num_features=64, reduction=16):
        super().__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N, C, H, W) => (N, C, 1, 1)
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    """Residual channel attention block, would not change (N, C, H, W)"""

    def __init__(self, num_features=64, reduction=16):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            CA(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    """Residual Group, would not change (N, C, H, W)"""

    def __init__(self, num_features=64, num_rcab=20, reduction=16):
        super().__init__()
        self.module = nn.Sequential(*[RCAB(num_features, reduction) for _ in range(num_rcab)])
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    """Residual Channel Attention Network, (N, C, H, W) => (N, C, H * scale, W * scale)"""

    def __init__(self, img_channel: int, scale: int, num_rg=10, num_features=64, num_rcab=20, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channel, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.conv3 = nn.Conv2d(num_features, img_channel, kernel_size=3, padding=1)

    def forward(self, x):  # input (N, img_channel, H, W)
        x = self.conv1(x)  # got (N, num_features, H, W)
        residual = x
        x = self.rgs(x)  # got (N, num_features, H, W)
        x = self.conv2(x)  # got (N, num_features, H, W)
        x += residual  # got (N, num_features, H, W)
        x = self.upscale(x)  # got (N, num_features, H * scale, W * scale)
        x = self.conv3(x)  # got (N, img_channel, H * scale, W * scale)
        return x


class RCANEncoder(nn.Module):
    """RCAN Encoder, (N, C, H, W) => (N, C=num_features, H, W)"""

    def __init__(self, img_channel: int, num_rg=10, num_features=64, num_rcab=20, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channel, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):  # input (N, img_channel, H, W)
        x = self.conv1(x)  # got (N, num_features, H, W)
        residual = x
        x = self.rgs(x)  # got (N, num_features, H, W)
        x = self.conv2(x)  # got (N, num_features, H, W)
        x += residual  # got (N, num_features, H, W)
        x = self.conv3(x)  # got (N, num_features, H * scale, W * scale)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.channel_adjust_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        return self.double_conv(x) + self.channel_adjust_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, None, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels, None, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # (C=in_channels, H, W) => (C=in_channels // 2, 2 * H, 2 * W)
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # size is (N, C, H, W)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 之前的pool操作可能导致shape的奇数元素缺一少二
        # padding是为了使x1的shape与x2的一样
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # (C=in_channels, H, W) => (C=in_channels // 2, 2 * H, 2 * W)
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels // 4, in_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # size is (N, C, H, W)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 之前的pool操作可能导致shape的奇数元素缺一少二
        # padding是为了使x1的shape与x2的一样
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PixelStack(nn.Module):
    def __init__(self, down_scale):
        super().__init__()
        self.down_scale = down_scale

    def squeeze2d(self, x):
        assert self.down_scale >= 1 and isinstance(self.down_scale,
                                                   int), f'Scale factor must be int, but got {self.down_scale}'
        if self.down_scale == 1:
            return x
        assert len(x.shape) == 4, f'Input shape should be (B, C, H, W)'
        B, C, H, W = x.shape
        assert H % self.down_scale == 0 and W % self.down_scale == 0, f'{(H, W)} must be divided by {self.down_scale} with no remainder'
        out = x.view(B, C, H // self.down_scale, self.down_scale, W // self.down_scale, self.down_scale)
        out = out.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = out.view(B, C * self.down_scale ** 2, H // self.down_scale, W // self.down_scale)
        return out

    def forward(self, x):
        return self.squeeze2d(x)


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_ReLU = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv_ReLU(x)


class UNetBased(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_conv = DoubleConv(in_channels, 64, None, kernel_size)
        self.down1 = Down(64, 128, kernel_size)
        self.down2 = Down(128, 256, kernel_size)
        self.down3 = Down(256, 512, kernel_size)
        self.down4 = Down(512, 1024, kernel_size)
        self.down5 = Down(1024, 2048, kernel_size)
        # self.down6 = Down(2048, 4096)
        # self.up6 = Up(4096, 2048)
        self.up5 = Up(2048, 1024, kernel_size)
        self.up4 = Up(1024, 512, kernel_size)
        self.up3 = Up(512, 256, kernel_size)
        self.up2 = Up(256, 128, kernel_size)
        self.up1 = Up(128, 64, kernel_size)

        self.stack1 = PixelStack(2)
        self.out_layer1 = ConvReLU(256, 64, kernel_size)
        self.stack2 = PixelStack(2)
        self.out_layer2 = ConvReLU(256, 64, kernel_size)
        self.stack3 = PixelStack(2)
        self.out_layer3 = ConvReLU(256, 64, kernel_size)
        self.out_layer4 = ConvReLU(64, 8, kernel_size)
        self.out_layer5 = ConvReLU(8, out_channels, kernel_size)

    def forward(self, x):
        x1 = self.in_conv(x)  # (1, 264, 264) => (64, 264, 264)
        x2 = self.down1(x1)  # => (128, 132, 132)
        x3 = self.down2(x2)  # => (256, 66, 66)
        x4 = self.down3(x3)  # => (512, 33, 33)
        x5 = self.down4(x4)  # => (1024, 16, 16)
        x6 = self.down5(x5)  # => (2048, 8, 8)
        # x7 = self.down6(x6)  # => (4096, 4, 4)
        # out = self.up6(x7, x6)  # => (2048, 8, 8)
        out = self.up5(x6, x5)  # => (1024, 16, 16)
        out = self.up4(out, x4)  # => (512, 33, 33)
        out = self.up3(out, x3)  # => (256, 66, 66)
        out = self.up2(out, x2)  # => (128, 132, 132)
        out = self.up1(out, x1)  # => (64, 264, 264)
        out = self.out_layer1(self.stack1(out))  # => (64, 132, 132)
        out = self.out_layer2(self.stack2(out))  # => (64, 66, 66)
        out = self.out_layer3(self.stack3(out))  # => (64, 33, 33)
        out = self.out_layer4(out)  # => (8, 33, 33)
        out = self.out_layer5(out)  # => (1, 33, 33)
        return out


class ResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.in_channels = opt['in_channels']
        self.out_channels = opt['out_channels']
        self.conv_kernel_size = opt['conv_kernel_size']
        self.num_pixel_stack_layer = opt['num_pixel_stack_layer']
        self.num_down_up = opt['num_down_up']
        assert self.num_down_up in (2, 3, 4, 5, 6)

        self.in_conv = ResDoubleConv(self.in_channels, 64, None, self.conv_kernel_size)

        self.down1 = ResDown(64, 128, self.conv_kernel_size)
        self.down2 = ResDown(128, 256, self.conv_kernel_size)
        if self.num_down_up >= 3:
            self.down3 = ResDown(256, 512, self.conv_kernel_size)
            if self.num_down_up >= 4:
                self.down4 = ResDown(512, 1024, self.conv_kernel_size)
                if self.num_down_up >= 5:
                    self.down5 = ResDown(1024, 2048, self.conv_kernel_size)
                    if self.num_down_up >= 6:
                        self.down6 = ResDown(2048, 4096, self.conv_kernel_size)
                        self.up6 = ResUp(4096, 2048, self.conv_kernel_size)
                    self.up5 = ResUp(2048, 1024, self.conv_kernel_size)
                self.up4 = ResUp(1024, 512, self.conv_kernel_size)
            self.up3 = ResUp(512, 256, self.conv_kernel_size)
        self.up2 = ResUp(256, 128, self.conv_kernel_size)
        self.up1 = ResUp(128, 64, self.conv_kernel_size)
        if self.num_pixel_stack_layer >= 1:
            self.down_sample = nn.Sequential(
                *[nn.Sequential(PixelStack(2), ConvReLU(256, 64, self.conv_kernel_size)) for _ in range(self.num_pixel_stack_layer)])
        self.out_layer1 = ConvReLU(64, 8, self.conv_kernel_size)
        self.out_layer2 = ConvReLU(8, self.out_channels, self.conv_kernel_size)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.num_down_up >= 3:
            x4 = self.down3(x3)
            if self.num_down_up >= 4:
                x5 = self.down4(x4)
                if self.num_down_up >= 5:
                    x6 = self.down5(x5)
                    if self.num_down_up >= 6:
                        x7 = self.down6(x6)
                        out = self.up6(x7, x6)
                        out = self.up5(out, x5)
                        out = self.up4(out, x4)
                        out = self.up3(out, x3)
                    else:
                        out = self.up5(x6, x5)
                        out = self.up4(out, x4)
                        out = self.up3(out, x3)
                        out = self.up2(out, x2)
                else:
                    out = self.up4(x5, x4)
                    out = self.up3(out, x3)
                    out = self.up2(out, x2)
            else:
                out = self.up3(x4, x3)
                out = self.up2(out, x2)
        else:
            out = self.up2(x3, x2)
        out = self.up1(out, x1)
        if self.num_pixel_stack_layer >= 1:
            out = self.down_sample(out)
        out = self.out_layer1(out)
        out = self.out_layer2(out)
        return out


class FFTResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.in_channels = opt['in_channels']
        self.out_channels = opt['out_channels']
        assert opt['encoder_channels'] % 2 == 0, 'encoder channels should be even'
        self.encoder_channels = opt['encoder_channels']
        self.conv_kernel_size = opt['conv_kernel_size']
        self.in_conv_relu1 = ConvReLU(in_channels=self.in_channels, out_channels=self.encoder_channels // 2, kernel_size=self.conv_kernel_size)
        self.in_conv_relu2 = ConvReLU(in_channels=self.in_channels, out_channels=self.encoder_channels // 2, kernel_size=self.conv_kernel_size)
        res_unet_opt = deepcopy(opt)
        res_unet_opt['in_channels'] = self.encoder_channels
        self.res_unet = ResUNet(res_unet_opt)

    def forward(self, x):
        x1 = self.in_conv_relu1(x)
        x2 = self.in_conv_relu2(x)
        x2 = torch.fft.fft2(x2)
        x2 = torch.abs(x2)
        x2 = torch.log10(1 + x2)
        ux = torch.cat([x1, x2], dim=-3)
        out = self.res_unet(ux)
        return out


class FFTRCANResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.in_channels = opt['in_channels']
        self.out_channels = opt['out_channels']
        assert opt['encoder_channels'] % 2 == 0, 'encoder channels should be even'
        self.encoder_channels = opt['encoder_channels']
        self.num_rg = opt['num_rg']
        self.num_rcab = opt['num_rcab']
        self.reduction = opt['reduction']
        self.in_rcan1 = RCANEncoder(img_channel=self.in_channels, num_rg=self.num_rg,
                                    num_features=self.encoder_channels // 2, num_rcab=self.num_rcab,
                                    reduction=self.reduction)
        self.in_rcan2 = RCANEncoder(img_channel=self.in_channels, num_rg=self.num_rg,
                                    num_features=self.encoder_channels // 2, num_rcab=self.num_rcab,
                                    reduction=self.reduction)
        res_unet_opt = deepcopy(opt)
        res_unet_opt['in_channels'] = self.encoder_channels
        self.res_unet = ResUNet(res_unet_opt)

    def forward(self, x):
        x1 = self.in_rcan1(x)
        x2 = self.in_rcan2(x)
        x2 = torch.fft.fft2(x2)
        x2 = torch.abs(x2)
        x2 = torch.log10(1 + x2)
        ux = torch.cat([x1, x2], dim=-3)
        out = self.res_unet(ux)
        return out


class FFTOnlyRCANResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.in_channels = opt['in_channels']
        self.out_channels = opt['out_channels']
        assert opt['encoder_channels'] % 2 == 0, 'encoder channels should be even'
        self.encoder_channels = opt['encoder_channels']
        self.num_rg = opt['num_rg']
        self.num_rcab = opt['num_rcab']
        self.reduction = opt['reduction']
        self.in_rcan = RCANEncoder(img_channel=self.in_channels * 2, num_rg=self.num_rg,
                                   num_features=self.encoder_channels, num_rcab=self.num_rcab,
                                   reduction=self.reduction)
        res_unet_opt = deepcopy(opt)
        res_unet_opt['in_channels'] = self.encoder_channels
        res_unet_opt['out_channels'] = 1
        self.res_unet = ResUNet(res_unet_opt)

    def forward(self, x):
        x = torch.fft.fft2(x)
        x1, x2 = torch.real(x), torch.imag(x)
        x1 = torch.sign(x1) * torch.log10(1 + torch.abs(x1))
        x2 = torch.sign(x2) * torch.log10(1 + torch.abs(x2))
        x = torch.cat([x1, x2], dim=-3)
        x = self.in_rcan(x)
        out = self.res_unet(x)
        return out


class FreqDomainRCANResUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.in_channels = opt['in_channels']
        self.out_channels = opt['out_channels']
        assert opt['encoder_channels'] % 2 == 0, 'encoder channels should be even'
        self.split_mode = opt['split_mode']
        self.encoder_channels = opt['encoder_channels']
        self.num_rg = opt['num_rg']
        self.num_rcab = opt['num_rcab']
        self.reduction = opt['reduction']
        self.in_rcan1 = RCANEncoder(img_channel=self.in_channels, num_rg=self.num_rg,
                                    num_features=self.encoder_channels // 2, num_rcab=self.num_rcab,
                                    reduction=self.reduction)
        self.in_rcan2 = RCANEncoder(img_channel=self.in_channels, num_rg=self.num_rg,
                                    num_features=self.encoder_channels // 2, num_rcab=self.num_rcab,
                                    reduction=self.reduction)
        res_unet_opt = deepcopy(opt)
        res_unet_opt['in_channels'] = self.encoder_channels
        self.res_unet = ResUNet(res_unet_opt)

        PSFsize = opt['psf_settings']['PSFsize']
        Pixelsize = opt['psf_settings']['Pixelsize']
        NA = opt['psf_settings']['NA']
        Lambda = opt['psf_settings']['Lambda']
        pos = torch.arange(-PSFsize / 2, PSFsize / 2, 1)
        Y, X = torch.meshgrid(pos, pos)
        k_r = (torch.sqrt(X * X + Y * Y)) / (PSFsize * Pixelsize)
        NA_constrain = torch.less(k_r, NA / Lambda)
        NA_closure = rectangular_closure(NA_constrain)
        self.mask_l = NA_closure[1] - NA_closure[0] + 1
        cut_h1 = (NA_constrain.shape[-2] - self.mask_l + 1) // 2
        cut_h2 = (NA_constrain.shape[-2] - self.mask_l) - cut_h1
        cut_w1 = (NA_constrain.shape[-1] - self.mask_l + 1) // 2
        cut_w2 = (NA_constrain.shape[-1] - self.mask_l) - cut_w1
        self.mask = NA_constrain[..., cut_h1:-cut_h2, cut_w1:-cut_w2]

    def get_input(self, x):
        x = torch.fft.fftshift(torch.fft.fft2(x))
        if self.split_mode == 'ap':
            x1, x2 = torch.log10(1.0 + torch.abs(x)), torch.angle(x)
        elif self.split_mode == 'ri':
            x1, x2 = torch.sign(x.real) * torch.log10(1.0 + torch.abs(x.real)), torch.sign(x.imag) * torch.log10(
                1.0 + torch.abs(x.imag))
        else:
            raise NotImplementedError
        cut_h1 = (x1.shape[-2] - self.mask_l + 1) // 2
        cut_h2 = (x1.shape[-2] - self.mask_l) - cut_h1
        cut_w1 = (x1.shape[-1] - self.mask_l + 1) // 2
        cut_w2 = (x1.shape[-1] - self.mask_l) - cut_w1
        x1 = x1[..., cut_h1:-cut_h2, cut_w1:-cut_w2]
        x1 = x1 * self.mask.to(x1.device)
        cut_h1 = (x2.shape[-2] - self.mask_l + 1) // 2
        cut_h2 = (x2.shape[-2] - self.mask_l) - cut_h1
        cut_w1 = (x2.shape[-1] - self.mask_l + 1) // 2
        cut_w2 = (x2.shape[-1] - self.mask_l) - cut_w1
        x2 = x2[..., cut_h1:-cut_h2, cut_w1:-cut_w2]
        # if self.split_mode == 'ap':
        #     x2 = x2.cpu().numpy()
        #     x2 = np.unwrap(np.unwrap(x2, axis=-1), axis=-2)
        #     x2 = torch.from_numpy(x2).to(x1.device)
        x2 = x2 * self.mask.to(x2.device)
        return x1, x2

    def forward(self, x):
        """x: (N, 1, H, W), out: (N, 1, h, w)"""
        x1, x2 = self.get_input(x)
        x1 = self.in_rcan1(x1)
        x2 = self.in_rcan2(x2)
        ux = torch.cat([x1, x2], dim=-3)
        out = self.res_unet(ux)
        out = torch.log(out + 1e-5)
        return out
