import argparse
import os
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, nearest_itpl, overlap, draw_text_on_image, \
    pickle_load


def test(opt):
    # pass parameter
    is_save = opt['testing']['is_save']
    show_kernel_code_psnr = opt['testing']['show_kernel_code_psnr']
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
    U_model = get_model(opt['U_Model'])
    F_model = get_model(opt['F_Model'])
    kernel_psnrs = []
    kernel_code_psnrs = []
    sr_psnrs = []

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                U_model.feed_data(data)
                U_model.test()
                pred_kernel = U_model.pred_kernel.squeeze(0).squeeze(0)
                pred_kernel_code = F_model.pca_encoder(pred_kernel.unsqueeze(0))
                gt_kernel = U_model.gt_kernel.squeeze(0)
                gt_kernel_code = F_model.pca_encoder(gt_kernel.unsqueeze(0))
                kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                if show_kernel_code_psnr:
                    offset = min(torch.min(pred_kernel_code).item(), torch.min(gt_kernel_code).item())
                    kernel_code_psnr = calculate_PSNR(pred_kernel_code - offset, gt_kernel_code - offset, max_val='auto')
                    kernel_code_psnrs.append(kernel_code_psnr)
                F_model.feed_data({'lr': data['lr'],
                                   'hr': data['hr'],
                                   'kernel_code': pred_kernel_code})
                F_model.test()
                sr_psnr = calculate_PSNR(F_model.hr, F_model.sr, max_val=1.0)
                sr_psnrs.append(sr_psnr)
                # LR->SR->HR
                result = torch.cat([normalization(F_model.lr), normalization(F_model.sr),
                                    normalization(F_model.hr)], dim=-1).squeeze(0).squeeze(0)
                show_size = (F_model.hr.shape[-2] // 4, F_model.hr.shape[-1] // 4)
                pred_kernel = nearest_itpl(pred_kernel, show_size)
                gt_kernel = nearest_itpl(gt_kernel, show_size)
                result = overlap(normalization(gt_kernel), result, (0, 0))
                result = overlap(normalization(pred_kernel), result, (gt_kernel.shape[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(F_model.hr.shape[-2] // 25, 16)
                if show_kernel_code_psnr:
                    draw_text_on_image(result, f'Kernel PSNR {kernel_psnr:5.2f}',
                                       (0, F_model.lr.shape[-2] - 2 * font_size), font_size, 65535)
                    draw_text_on_image(result, f'Code PSNR {kernel_code_psnr:5.2f}',
                                       (0, F_model.lr.shape[-2] - font_size), font_size, 65535)
                draw_text_on_image(result, f'PSNR {sr_psnr:5.2f}', (result.width // 3, 0), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (result.width // 3 * 2, 0), font_size, 65535)
                if is_save:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                pbar.update(1)
    print(f'avg psnr: sr={np.mean(sr_psnrs):5.2f}, kernel={np.mean(kernel_psnrs):5.2f}, code={np.mean(kernel_code_psnrs):5.2f}')


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