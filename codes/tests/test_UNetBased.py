import argparse
import os
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from scipy.io import savemat

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, nearest_itpl, overlap, draw_text_on_image, \
    pickle_load


def test(opt):
    # pass parameter
    is_save = opt['testing']['is_save']
    save_kernel = opt['testing']['save_kernel']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if is_save and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up data loader
    if opt['testing']['preload_data'] is not None:
        test_loader = pickle_load(opt['testing']['preload_data'])
        print(f'load test data from {opt["testing"]["preload_data"]}')
    else:
        test_loader = get_dataloader(opt['test_data'])
        print('generate test data on the fly')

    # set up model
    model = get_model(opt['model'])
    mat_data = {'UNetBased_pred_kernels': [], 'names': []}
    kernel_psnrs = []

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                model.feed_data(data)
                model.test()
                if save_kernel:
                    mat_data['UNetBased_pred_kernels'].append(model.pred_kernel.squeeze(0).squeeze(0).detach().cpu().numpy())
                    mat_data['names'].append(data['name'][0])
                result = normalization(model.lr).squeeze(0).squeeze(0)
                show_size = (model.lr.shape[-2] // 4, model.lr.shape[-1] // 4)
                pred_kernel = model.pred_kernel.squeeze(0).squeeze(0)
                gt_kernel = model.gt_kernel.squeeze(0)
                kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                pred_kernel = nearest_itpl(pred_kernel, show_size)
                gt_kernel = nearest_itpl(gt_kernel, show_size)
                result = overlap(normalization(gt_kernel), result, (0, 0))
                result = overlap(normalization(pred_kernel), result, (gt_kernel.shape[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(model.lr.shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {kernel_psnr:5.2f}',
                                   (0, model.lr.shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (0, model.lr.shape[-2] - font_size), font_size, 65535)
                if is_save:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                pbar.update(1)
    if save_kernel:
        savemat(os.path.join(save_dir, 'pred_kernels.mat'), mat_data)
    print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')


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
