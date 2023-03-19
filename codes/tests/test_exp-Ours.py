import argparse
import os

import torch
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
from tqdm import tqdm

from models import get_model
from utils.universal_util import read_yaml, normalization, overlap


def test(opt):
    # pass parameter
    lr_path = opt['testing']['lr_path']
    save_img = opt['testing']['save_img']
    save_mat = opt['testing']['save_mat']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if (save_img['sr'] or save_img['kernel'] or save_img['sr_kernel'] or save_mat) and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up model
    U_model = get_model(opt['U_Model'])
    F_model = get_model(opt['F_Model'])

    # set up recorder
    pred_kernels = []
    names = []

    # start testing
    with tqdm(desc=f'testing', total=len(os.listdir(lr_path)), unit='img') as pbar:
        with torch.no_grad():
            for name in os.listdir(lr_path):
                img = transforms.ToTensor()(Image.open(os.path.join(lr_path, name))).float().unsqueeze(0) / 65535.0
                U_model.feed_data({'lr': img, 'kernel': torch.rand((1, 1, 33, 33))})
                U_model.test()
                kernel_for_sr = U_model.pred_kernel.squeeze(0)
                F_model.feed_data({'lr': img,
                                   'hr': torch.rand(size=img.shape),
                                   'kernel': kernel_for_sr})
                F_model.test()

                pred_kernel = kernel_for_sr.squeeze(0)
                pred_kernels.append(pred_kernel.detach().cpu().numpy())
                names.append(name)

                result_sr = normalization(F_model.sr).squeeze(0).squeeze(0)
                result_kernel = normalization(pred_kernel).squeeze(0).squeeze(0)
                result_all = overlap(result_kernel, result_sr, (0, 0))
                result_sr = transforms.ToPILImage()((result_sr * 65535).to(torch.int32))
                result_kernel = transforms.ToPILImage()((result_kernel * 65535).to(torch.int32))
                result_all = transforms.ToPILImage()((result_all * 65535).to(torch.int32))

                if save_img['sr']:
                    result_sr.save(os.path.join(save_dir, name))
                if save_img['kernel']:
                    result_kernel.save(os.path.join(save_dir, name.replace('.png', '_k.png')))
                if save_img['sr_kernel']:
                    result_all.save(os.path.join(save_dir, name.replace('.png', '_sr-k.png')))
                pbar.update(1)
    if save_mat:
        savemat(os.path.join(save_dir, 'results.mat'), {'UNetBased_pred_kernels': pred_kernels,
                                                        'names': names})


def main():
    """please make sure that the pwd is .../PsfPred rather than .../PsfPred/codes/tests"""
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/test_something.yaml')
    args = prog.parse_args()

    # start train
    test(read_yaml(args.opt))


if __name__ == '__main__':
    main()
