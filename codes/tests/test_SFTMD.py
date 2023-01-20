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
    show_kernel = opt['testing']['show_kernel']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if is_save and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up data loader
    if opt['testing']['preload_data'] is not None:
        test_loader = pickle_load(opt['testing']['preload_data'])
    else:
        test_loader = get_dataloader(opt['test_data'])

    # set up model
    model = get_model(opt['model'])
    sr_psnrs = []

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                model.feed_data(data)
                model.test()
                sr_psnr = calculate_PSNR(model.hr, model.sr, max_val=1.0)
                sr_psnrs.append(sr_psnr)
                # LR->SR->HR
                result = torch.cat([normalization(model.lr), normalization(model.sr),
                                    normalization(model.hr)], dim=-1).squeeze(0).squeeze(0)
                if show_kernel:
                    show_size = (model.hr.shape[-2] // 4, model.hr.shape[-1] // 4)
                    kernel = nearest_itpl(model.kernel.squeeze(0), show_size)
                    result = overlap(normalization(kernel), result, (0, 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(model.hr.shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {sr_psnr:5.2f}', (result.width // 3, 0), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (result.width // 3 * 2, 0), font_size, 65535)
                if is_save:
                    result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                pbar.update(1)
    print(f'avg psnr: sr={np.mean(sr_psnrs):5.2f}')


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
