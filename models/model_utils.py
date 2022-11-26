from os import listdir
from os.path import join
from torch import load
from general_utils import get_the_latest_file
from models.unet_based import FFTResUNet, FFTRCANResUNet, FFTOnlyRCANResUNet, FreqDomainRCANResUNet
from models.MANet import MANet_s1


def get_model(opt):
    if opt['network'] == 'FFTResUNet':
        return FFTResUNet(opt)
    elif opt['network'] == 'FFTRCANResUNet':
        return FFTRCANResUNet(opt)
    elif opt['network'] == 'FFTOnlyRCANResUNet':
        return FFTOnlyRCANResUNet(opt)
    elif opt['network'] == 'FreqDomainRCANResUNet':
        return FreqDomainRCANResUNet(opt)
    elif opt['network'] == 'MANet_s1':
        return MANet_s1(opt)
    else:
        raise NotImplementedError


def restore_checkpoint(obj, root, mode='latest', postfix='_net.pth'):
    if mode == 'latest':
        path_step = get_the_latest_file(root, postfix)
        path = path_step['path']
        step = path_step['step']
    elif mode == 'best':
        best_file = [file for file in listdir(root) if (file.endswith(postfix) and 'best' in file)]
        if len(best_file) == 0:
            raise FileNotFoundError
        if len(best_file) > 1:
            raise RuntimeError('there are more than one best checkpoints')
        path = join(root, best_file[0])
        step = int(best_file[0].replace(postfix, '').replace('best_', ''))
    else:
        raise NotImplementedError
    obj.load_state_dict(load(path))
    return {'path': path,
            'step': step}
