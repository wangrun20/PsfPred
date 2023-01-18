from torch.utils.data import DataLoader

from datasets.hr_lr_kernel_from_BioSR import HrLrKernelFromBioSR


def get_dataloader(opt):
    if opt['name'] == 'HrLrKernelFromBioSR':
        data_set = HrLrKernelFromBioSR(opt)
    else:
        raise NotImplementedError
    data_loader = DataLoader(data_set, **opt['loader_settings'])
    return data_loader


def main():
    opt = {'name': 'HrLrKernelFromBioSR',
           'is_train': True,
           'gpu_id': None,  # None for cpu
           'repeat': 20,
           'img_filter': {'img_root': '../../../BioDatasets/BioSR/Mixed',
                          'structure_selected': [1, 2, 3, 4],
                          'included_idx': [11, 100]},
           'hr_cropping': {'mode': 'random',  # random | constant | scanning
                           'center_pos': [-1, -1],  # [H, W], for constant
                           'scanning_shape': [-1, -1],  # [H, W], for scanning
                           'hr_size': [264, 264]},  # [H, W]
           'img_signal': [100, 1000],
           'is_norm_hr': False,
           'is_norm_lr': False,
           'is_norm_k': False,
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
           'sup_phaseZ': [4, 8],  # all | [begin, end]
           'padding': {'mode': 'circular',  # constant | reflect | replicate | circular
                       'value': 0},  # for constant mode
           'loader_settings': {'batch_size': 4,
                               'shuffle': True,
                               'num_workers': 7,
                               'pin_memory': False,
                               'drop_last': True}}
    dataloader = get_dataloader(opt)
    print(len(dataloader), len(dataloader.dataset),
          dataloader.dataset[0]['hr'].shape, dataloader.dataset[0]['lr'].shape,
          dataloader.dataset[0]['kernel'].shape, dataloader.dataset[0]['name'],
          dataloader.dataset[0]['phaseZ'].shape, dataloader.dataset[0]['img_signal'])


if __name__ == '__main__':
    main()
