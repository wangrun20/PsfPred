import math

import torch
from torch.nn.functional import pad
from zernike_psf import ZernikePSFGenerator
from utils import get_phaseZ, calculate_PSNR, save_gray_img


def main():
    opt = {'device': torch.device('cpu'),
           'kernel_size': 33,
           'NA': 1.35,
           'Lambda': 0.525,
           'RefractiveIndex': 1.33,
           'SigmaX': 2.0,
           'SigmaY': 2.0,
           'Pixelsize': 0.0313,
           'nMed': 1.33}
    psf_gen = ZernikePSFGenerator(opt=opt)
    mask = torch.linspace(0, 24, 25).unsqueeze(0)
    # phaseZ = get_phaseZ(opt={'idx_start': 4, 'num_idx': 15, 'mode': 'gaussian', 'std': 0.125, 'bound': 1.0},
    #                     batch_size=1, device=torch.device('cpu'))
    phaseZ = torch.linspace(-1.0, 1.0, 25).unsqueeze(0) * (4 <= mask) * (mask < 19)
    phaseZ_1to5 = phaseZ * (4 <= mask) * (mask < 9)
    phaseZ_6to10 = phaseZ * (9 <= mask) * (mask < 14)
    phaseZ_11to15 = phaseZ * (14 <= mask) * (mask < 19)

    k = psf_gen.generate_PSF(phaseZ, blur=False)
    k1 = psf_gen.generate_PSF(phaseZ_1to5, blur=False)
    k2 = psf_gen.generate_PSF(phaseZ_6to10, blur=False)
    k3 = psf_gen.generate_PSF(phaseZ_11to15, blur=False)

    def my_conv(img, kernel):
        assert len(img.shape) == len(kernel.shape) == 3 and img.shape[0] == kernel.shape[0] == 1
        img = pad(img.unsqueeze(0), pad=(kernel.shape[-2] // 2, kernel.shape[-2] - kernel.shape[-2] // 2 - 1
                                         , kernel.shape[-1] // 2, kernel.shape[-1] - kernel.shape[-1] // 2 - 1),
                  mode='circular')
        return torch.nn.functional.conv2d(img, kernel.unsqueeze(0)).squeeze(0)

    k123 = my_conv(my_conv(k1, k2), k3)
    psnr = calculate_PSNR(k, k123, max_val=max(torch.max(k).item(), torch.max(k123).item()))
    # save_gray_img(k.squeeze(0), './k.png')
    # save_gray_img(k1.squeeze(0), './k1.png')
    # save_gray_img(k2.squeeze(0), './k2.png')
    # save_gray_img(k3.squeeze(0), './k3.png')
    # save_gray_img(k123.squeeze(0), './k123.png')
    heat_map = (k123 == k)
    print(psnr)


if __name__ == '__main__':
    g = lambda x: math.log(x) - 2./x - 1./x**2
    g1 = lambda x: 1./x + 2./x**2 + 2./x**3
    g2 = lambda x: -1./x**2 - 4./x**3 - 6./x**4
    g3 = lambda x: 2./x**3 + 12./x**4 + 24./x**5
    phi2 = lambda x: g2(x)/g1(x) + g(x)*g3(x)/(g1(x)**2) - 2.*g(x)*(g2(x)**2)/(g1(x)**3)
    result = []
    for i in range(101):
        xx = 2. + i / 100
        result.append((xx, g(xx), g1(xx), g2(xx), g3(xx), phi2(xx)))
    print(result)
    main()
