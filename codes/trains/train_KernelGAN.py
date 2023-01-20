import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

from models.KernelGAN import KernelGAN, Learner
from datasets.KernelGAN_data_generator import DataGenerator
from utils.universal_util import read_yaml, pickle_load, calculate_PSNR


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
        for file_name in os.listdir(opt['input_image_root']):
            opt['input_image_path'] = os.path.join(opt['input_image_root'], file_name)
            opt['img_name'] = os.path.splitext(opt['input_image_path'])[0]
            train(opt)
    else:
        try:
            from datasets.hr_lr_kernel_from_BioSR import HrLrKernelFromBioSR
            testset = pickle_load(opt['preload_data']).dataset
            for i in range(len(testset)):
                opt['input_image_path'] = testset[i]['name'] + '.png'
                opt['img_name'] = testset[i]['name']
                opt['input_image'] = torch.permute(testset[i]['lr'], (1, 2, 0)).contiguous().cpu().numpy()
                pred_kernel = train(opt)
                del opt['input_image']
                kernel_psnr = calculate_PSNR(pred_kernel, testset[i]['kernel'], max_val='auto')
                print(kernel_psnr)
                kernel_psnrs.append(kernel_psnr)
            print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')
        except KeyboardInterrupt:
            print(f'avg psnr: kernel={np.mean(kernel_psnrs):5.2f}')


if __name__ == '__main__':
    main()
