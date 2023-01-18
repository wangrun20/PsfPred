import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms
import scipy.io as sio
from scipy.ndimage import measurements, interpolation

import loss_functions
from networks import get_network


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def save_final_kernel(k_2, conf):
    """saves the final kernel and the analytic kernel to the results folder"""
    sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x2.mat' % os.path.basename(conf.img_name)),
                {'Kernel': k_2})
    k = torch.from_numpy(k_2)
    k = (k - torch.min(k)) / (torch.max(k) - torch.min(k))
    k = torchvision.transforms.ToPILImage()(k)
    k.save(os.path.join(conf.output_dir_path, '%s_kernel_x2.png' % os.path.basename(conf.img_name)))
    if conf.X4:
        k_4 = analytic_kernel(k_2)
        sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x4.mat' % conf.img_name), {'Kernel': k_4})


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting, so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th the largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def post_process_k(k, n):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = move2cpu(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k


class KernelGAN(object):
    # Constraint co-efficients
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0

    def __init__(self, opt):
        # Acquire configuration
        self.opt = opt

        # Define the GAN
        self.G = get_network(opt['net_G']).cuda()
        self.D = get_network(opt['net_D']).cuda()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, opt['net_G']['input_crop_size'], opt['net_G']['input_crop_size']).cuda()
        self.d_input = torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(opt['G_kernel_size'], opt['G_kernel_size']).cuda()

        # Losses
        self.GAN_loss_layer = loss_functions.GANLoss(d_last_layer_size=self.d_output_shape).cuda()
        self.bicubic_loss = loss_functions.DownScaleLoss(scale_factor=opt['net_G']['scale_factor']).cuda()
        self.sum2one_loss = loss_functions.SumOfWeightsLoss().cuda()
        self.boundaries_loss = loss_functions.BoundariesLoss(k_size=opt['G_kernel_size']).cuda()
        self.centralized_loss = loss_functions.CentralizedLoss(k_size=opt['G_kernel_size'],
                                                               scale_factor=opt['net_G']['scale_factor']).cuda()
        self.sparse_loss = loss_functions.SparsityLoss().cuda()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt['g_lr'], betas=(opt['beta1'], 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt['d_lr'], betas=(opt['beta1'], 0.999))

        print('*' * 60 + '\nSTARTED KernelGAN on: \"%s\"...' % opt['input_image_path'])

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.parameters()):
            curr_k = F.conv2d(delta, w, padding=self.opt['G_kernel_size'] - 1) if ind == 0 else F.conv2d(curr_k, w)
        self.curr_k = curr_k.squeeze().flip([0, 1])

    def train(self, g_input, d_input):
        self.set_input(g_input, d_input)
        self.train_g()
        self.train_d()

    def set_input(self, g_input, d_input):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        g_pred = self.G.forward(self.g_input)
        # Pass Generators output through Discriminator
        d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        total_loss_g = loss_g + self.calc_constraints(g_pred)
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()

    def calc_constraints(self, g_pred):
        # Calculate K which is equivalent to G
        self.calc_curr_k()
        # Calculate constraints
        self.loss_bicubic = self.bicubic_loss.forward(g_input=self.g_input, g_output=g_pred)
        loss_boundaries = self.boundaries_loss.forward(kernel=self.curr_k)
        loss_sum2one = self.sum2one_loss.forward(kernel=self.curr_k)
        loss_centralized = self.centralized_loss.forward(kernel=self.curr_k)
        loss_sparse = self.sparse_loss.forward(kernel=self.curr_k)
        # Apply constraints co-efficients
        return self.loss_bicubic * self.lambda_bicubic + loss_sum2one * self.lambda_sum2one + \
            loss_boundaries * self.lambda_boundaries + loss_centralized * self.lambda_centralized + \
            loss_sparse * self.lambda_sparse

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)
        d_pred_fake = self.D.forward((g_output + torch.randn_like(g_output) / 255.).detach())
        # Calculate discriminator loss
        loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def finish(self):
        final_kernel = post_process_k(self.curr_k, n=self.opt['n_filtering'])
        save_final_kernel(final_kernel, self.opt)
        print('KernelGAN estimation complete!')
        if self.opt['do_ZSSR']:
            # run_zssr(final_kernel, self.opt)
            raise NotImplementedError('ZSSR has not been implemented')
        print('FINISHED RUN (see --%s-- folder)\n' % self.opt['output_dir_path'] + '*' * 60 + '\n\n')


def main():
    gan = KernelGAN(opt={'input_image_path': None,
                         'img_name': None,
                         'img_max_val': 65535.0,
                         'noise_scale': 40,
                         'output_dir_path': None,
                         'real_image': False,
                         'G_kernel_size': 33,
                         'net_G': {'name': 'Generator',
                                   'G_structure': (7, 7, 5, 5, 5, 3, 3, 3, 3, 1, 1, 1),
                                   'G_chan': 64,
                                   'scale_factor': 1.0,
                                   'input_crop_size': 64,
                                   'init': None},
                         'net_D': {'name': 'Discriminator',
                                   'img_channel': 3,
                                   'D_chan': 64,
                                   'D_kernel_size': 7,
                                   'D_n_layers': 7,
                                   'input_crop_size': 64,
                                   'init': None},
                         'gpu_id': 0,
                         'max_iter': 3000,
                         'n_filtering': 40,
                         'X4': False,
                         'beta1': 0.5,
                         'g_lr': 0.0002,
                         'd_lr': 0.0002,
                         'do_ZSSR': False})
    print(gan)


if __name__ == '__main__':
    main()
