import os
import random
from PIL import Image
import math
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

from utils.universal_util import random_rotate_crop_flip, add_poisson_gaussian_noise, get_phaseZ, normalization
from utils.zernike_psf import ZernikePSFGenerator


class HrLrKernelFromBioSR(Dataset):
    def __init__(self, opt):
        """
        img names in img_root should be img_xx_y.png, such like img_13_4.png, img_29_3.png
        where y is structure type (CCPs | ER | Microtubules | F-actin), x is index
        """
        super().__init__()
        self.is_train = opt['is_train']
        self.repeat = opt['repeat'] if opt['repeat'] is not None else 1
        self.device = torch.device('cpu') if opt['gpu_id'] is None else torch.device('cuda', opt['gpu_id'])
        self.img_root = opt['img_filter']['img_root']
        self.structure_selected = tuple(opt['img_filter']['structure_selected'])
        self.included_idx = tuple(range(opt['img_filter']['included_idx'][0], opt['img_filter']['included_idx'][1] + 1))
        self.hr_crop = opt['hr_crop']
        self.hr_size = tuple(opt['hr_crop']['hr_size'])
        self.scale = opt['scale']
        self.img_signal = opt['img_signal']
        self.phaseZ_settings = opt['psf_settings']['phaseZ']
        self.sup_phaseZ = opt['sup_phaseZ']
        opt['psf_settings']['device'] = self.device
        self.psf_gen = ZernikePSFGenerator(opt=opt['psf_settings'])
        self.padding = opt['padding']

        assert self.hr_size[0] % self.scale == 0 and self.hr_size[1] % self.scale == 0

        all_gt = os.listdir(self.img_root)
        all_gt.sort()
        self.names = [os.path.splitext(file)[0] for file in all_gt
                      if (file.endswith('.png')
                          and (int(file.split('_')[2].replace('.png', '')) in self.structure_selected)
                          and (int(file[4:6]) in self.included_idx))]

        if not self.is_train:
            # hrs are fixed when testing
            self.hrs = torch.cat([self.get_aug_hr(i // self.repeat) for i in range(len(self))], dim=-3)
            self.test_phaseZs = get_phaseZ(self.phaseZ_settings, batch_size=len(self), device=self.device)
            self.test_kernels = self.psf_gen.generate_PSF(phaseZ=self.test_phaseZs)

    def __len__(self):
        return len(self.names) * self.repeat * (self.hr_crop['scan_shape'][0] * self.hr_crop['scan_shape'][1]
                                                if self.hr_crop['mode'] == 'scan' else 1)

    def get_aug_hr(self, idx):
        """
        :return: GT image, in shape of self.hr_size, with data augmentation
        """
        name = self.names[idx % len(self.names)] if self.hr_crop['mode'] == 'scan' else self.names[idx]
        img = transforms.ToTensor()(Image.open(os.path.join(self.img_root, name + '.png'))).float()
        if self.hr_crop['mode'] == 'random':
            fill = 0
            while True:  # avoid dark regions
                hr = random_rotate_crop_flip(img, self.hr_size, fill)
                structure = int(name.split('_')[2].replace('.png', ''))
                if structure in (1,) and hr.mean() >= 60:
                    break  # CCPs
                elif structure in (2, 3, 4) and (torch.max(hr) - torch.min(hr)).item() >= 10000:
                    break  # ER, Microtubules, F-actin
                elif structure not in (1, 2, 3, 4):
                    raise NotImplementedError
        elif self.hr_crop['mode'] == 'constant':
            center_pos = self.hr_crop['center_pos']
            bound = ((center_pos[0] - (self.hr_size[0] // 2), center_pos[0] + self.hr_size[0] - (self.hr_size[0] // 2)),
                     (center_pos[1] - (self.hr_size[1] // 2), center_pos[1] + self.hr_size[1] - (self.hr_size[1] // 2)))
            hr = img[:, bound[0][0]:bound[0][1], bound[1][0]:bound[1][1]]
        elif self.hr_crop['mode'] == 'scan':
            start_h = int((img.height - self.hr_size[0]) / self.hr_crop['scan_shape'][0] *
                          ((idx // len(self.names)) % self.hr_crop['scan_shape'][0]))
            start_w = int((img.width - self.hr_size[1]) / self.hr_crop['scan_shape'][1] *
                          ((idx // len(self.names)) // self.hr_crop['scan_shape'][0]))
            hr = img[:, start_h:start_h + self.hr_size[0], start_w:start_w + self.hr_size[1]]
        elif self.hr_crop['mode'] is None:
            hr = img
        else:
            raise NotImplementedError('undefined mode')
        return hr.to(self.device)  # (C, H, W), 0~65535

    def __getitem__(self, index):
        idx = index // self.repeat
        if self.is_train:
            hr = self.get_aug_hr(idx)
            phaseZ = get_phaseZ(self.phaseZ_settings, batch_size=1, device=self.device)
            kernel = self.psf_gen.generate_PSF(phaseZ=phaseZ)
        else:
            hr = self.hrs[idx:idx + 1, :, :]
            phaseZ = self.test_phaseZs[idx:idx + 1, :]
            kernel = self.test_kernels[idx:idx + 1, :, :]
        assert kernel.shape[-2] % 2 == 1 and kernel.shape[-1] % 2 == 1, 'kernel shape should be odd'

        pad = (kernel.shape[-2] // 2,) * 2 + (kernel.shape[-1] // 2,) * 2
        if self.padding['mode'] == "circular":
            lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode=self.padding['mode']), kernel.unsqueeze(0)).squeeze(0)
        else:
            lr = F.conv2d(F.pad(hr.unsqueeze(0), pad=pad, mode=self.padding['mode'], value=self.padding['value']),
                          kernel.unsqueeze(0)).squeeze(0)
        img_signal = 10.0 ** random.uniform(math.log10(self.img_signal[0]), math.log10(self.img_signal[-1]))
        lr = add_poisson_gaussian_noise(lr, level=img_signal)
        lr = lr[:, ::self.scale, ::self.scale]

        if self.sup_phaseZ == 'all':
            pass
        else:
            cut_phaseZ = torch.zeros(size=phaseZ.shape, dtype=phaseZ.dtype, device=phaseZ.device)
            cut_phaseZ[..., self.sup_phaseZ[0]:self.sup_phaseZ[-1] + 1] = \
                phaseZ[..., self.sup_phaseZ[0]:self.sup_phaseZ[-1] + 1]
            kernel = self.psf_gen.generate_PSF(phaseZ=cut_phaseZ)

        if self.hr_crop['mode'] == 'scan':
            name = self.names[idx % len(self.names)] + f'_part{idx // len(self.names)}' + \
                   f'_{(index % self.repeat) + 1}'
        else:
            name = self.names[idx] + f'_{(index % self.repeat) + 1}'

        hr = hr / 65535.0
        lr = lr / 65535.0
        return {'hr': hr,  # (C, H, W), [0, 1]
                'lr': lr,  # (C, H, W), [0, 1]
                'kernel': kernel.squeeze(0),  # (H, W), sum up to 1.0
                'name': name,  # str, without postfix '.png'
                'phaseZ': phaseZ,  # (1, 25)
                'img_signal': img_signal}  # float


def main():
    opt = {'name': 'HrLrKernelFromBioSR',
           'is_train': True,
           'gpu_id': None,  # None for cpu
           'repeat': None,
           'img_filter': {'img_root': '../../../BioDatasets/BioSR/Mixed',
                          'structure_selected': [1, 2, 3, 4],
                          'included_idx': [11, 100]},
           'hr_crop': {'mode': 'random',  # random | constant | scan
                       'center_pos': [-1, -1],  # [H, W], for constant
                       'scan_shape': [-1, -1],  # [H, W], for scan
                       'hr_size': [264, 264]},  # [H, W]
           'scale': 2,
           'img_signal': [100, 1000],
           'psf_settings': {'kernel_size': 33,
                            'NA': 1.35,
                            'Lambda': 0.525,
                            'RefractiveIndex': 1.33,
                            'SigmaX': 2.0,
                            'SigmaY': 2.0,
                            'Pixelsize': 0.0313,
                            'nMed': 1.33,
                            'phaseZ': {'idx_start': 4,
                                       'num_idx': 15,
                                       'mode': 'gaussian',  # gaussian | uniform
                                       'std': 0.125,  # for gaussian
                                       'bound': 1.0}},  # for gaussian and uniform
           'sup_phaseZ': 'all',  # all | [begin, end]
           'padding': {'mode': 'circular',  # constant | reflect | replicate | circular
                       'value': -1},  # for constant mode
           'loader_settings': {'batch_size': 4,
                               'shuffle': True,
                               'num_workers': 3,
                               'pin_memory': False,
                               'drop_last': True}}
    data_set = HrLrKernelFromBioSR(opt)
    data_loader = torch.utils.data.DataLoader(data_set, **opt['loader_settings'])
    print(len(data_loader), len(data_set),
          data_set[0]['hr'].shape, data_set[0]['lr'].shape,
          data_set[0]['kernel'].shape, data_set[0]['name'],
          data_set[0]['phaseZ'].shape, data_set[0]['img_signal'])


if __name__ == '__main__':
    main()
