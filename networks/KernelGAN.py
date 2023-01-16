import torch
import torch.nn as nn
import numpy as np


def swap_axis(im):
    """Swap axis of a tensor from a 3 channel tensor to a batch of 3-single channel and vise-versa"""
    return im.transpose(0, 1) if type(im) == torch.Tensor else np.moveaxis(im, 0, 1)


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        struct = opt.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=opt.G_chan, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(opt.G_chan, opt.G_chan, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=opt.G_chan, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / opt.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(
            torch.ones([1, 1, opt.input_crop_size, opt.input_crop_size]))).shape[-1]
        self.forward_shave = int(opt.input_crop_size * opt.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)


class Discriminator(nn.Module):

    def __init__(self, opt):
        super().__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(opt.img_channel, opt.D_chan,
                                                            kernel_size=opt.D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, opt.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(opt.D_chan, opt.D_chan,
                                                               kernel_size=1, bias=True)),
                              nn.BatchNorm2d(opt.D_chan),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(opt.D_chan, 1,
                                                                          kernel_size=1, bias=True)),
                                         nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = opt.input_crop_size - self.forward(torch.FloatTensor(
            torch.ones([1, opt.img_channel, opt.input_crop_size, opt.input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)


def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
