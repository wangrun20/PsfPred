import argparse
import os
import torch
from torchvision import transforms
from tqdm import tqdm
from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, nearest_itpl, overlap, draw_text_on_image


def test(opt):
    # pass parameter
    save_dir = opt['testing']['save_dir']

    # mkdir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up data loader
    test_loader = get_dataloader(opt['test_data'])

    # set up model
    model = get_model(opt['model'])

    # start testing
    with tqdm(desc=f'testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                model.feed_data(data)
                model.test()
                result = normalization(model.lr).squeeze(0).squeeze(0)
                show_size = (model.lr.shape[-2] // 4, model.lr.shape[-1] // 4)
                pred_kernel = model.pred_kernel.squeeze(0).squeeze(0)
                gt_kernel = model.gt_kernel.squeeze(0)
                kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                pred_kernel = nearest_itpl(pred_kernel, show_size)
                gt_kernel = nearest_itpl(gt_kernel, show_size)
                result = overlap(normalization(gt_kernel), result, (0, 0))
                result = overlap(normalization(pred_kernel), result, (gt_kernel.shape[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(model.lr.shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {kernel_psnr:5.2f}',
                                   (0, model.lr.shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, data['name'][0], (0, model.lr.shape[-2] - font_size), font_size, 65535)
                result.save(os.path.join(save_dir, data['name'][0] + '.png'))
                pbar.update(1)


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
