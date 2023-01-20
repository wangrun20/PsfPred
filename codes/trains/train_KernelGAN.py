import argparse
import os
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np

from models.KernelGAN import KernelGAN, Learner
from datasets.KernelGAN_data_generator import DataGenerator
from utils.universal_util import read_yaml, pickle_load, calculate_PSNR, normalization, nearest_itpl, overlap, \
    draw_text_on_image


def train(opt):
    gan = KernelGAN(opt)
    learner = Learner()
    data = DataGenerator(opt, gan)
    for iteration in tqdm(range(opt['max_iters']), ncols=60):
        [g_in, d_in] = data[iteration]
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    no_shift_kernel, _ = gan.finish()
    return no_shift_kernel  # sum up to 1.0


def main():
    """please make sure that the pwd is .../PsfPred rather than .../PsfPred/codes/trains"""
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/train_something.yaml')
    args = prog.parse_args()

    # start train
    opt = read_yaml(args.opt)
    kernel_psnrs = []
    if not os.path.exists(opt['output_dir_path']):
        os.mkdir(opt['output_dir_path'])
    if opt['preload_data'] is None:
        print(f'load test data from {opt["input_image_root"]}')
        for file_name in os.listdir(opt['input_image_root']):
            opt['input_image_path'] = os.path.join(opt['input_image_root'], file_name)
            opt['img_name'] = os.path.splitext(opt['input_image_path'])[0]
            train(opt)
    else:
        try:
            from datasets.hr_lr_kernel_from_BioSR import HrLrKernelFromBioSR
            testset = pickle_load(opt['preload_data']).dataset
            print(f'load test data from {opt["preload_data"]}')
            for i in range(len(testset)):
                opt['input_image_path'] = testset[i]['name'] + '.png'
                opt['img_name'] = testset[i]['name']
                opt['input_image'] = torch.permute(testset[i]['lr'], (1, 2, 0)).contiguous().cpu().numpy()
                pred_kernel = torch.from_numpy(train(opt))
                result = normalization(testset[i]['lr']).squeeze(0)
                show_size = (testset[i]['lr'].shape[-2] // 4, testset[i]['lr'].shape[-1] // 4)
                gt_kernel = testset[i]['kernel']
                kernel_psnr = calculate_PSNR(pred_kernel, testset[i]['kernel'], max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                pred_kernel = nearest_itpl(pred_kernel, show_size)
                gt_kernel = nearest_itpl(gt_kernel, show_size)
                result = overlap(normalization(gt_kernel), result, (0, 0))
                result = overlap(normalization(pred_kernel), result, (gt_kernel.shape[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(testset[i]['lr'].shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {kernel_psnr:5.2f}',
                                   (0, testset[i]['lr'].shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, testset[i]['name'],
                                   (0, testset[i]['lr'].shape[-2] - font_size), font_size, 65535)
                result.save(os.path.join(opt['output_dir_path'], testset[i]['name'] + '.png'))
                print(f'kernel psnr={kernel_psnr}\n')
            print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')
        except KeyboardInterrupt:
            print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')


if __name__ == '__main__':
    main()
