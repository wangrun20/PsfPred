from torch.utils.data import DataLoader

from datasets.hr_lr_kernel_from_BioSR import HrLrKernelFromBioSR


def get_dataloader(opt):
    if opt['name'] == 'HrLrKernelFromBioSR':
        data_set = HrLrKernelFromBioSR(opt)
    else:
        raise NotImplementedError
    data_loader = DataLoader(data_set, **opt['loader_settings'])
    return data_loader
