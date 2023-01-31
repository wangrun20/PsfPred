import argparse
import os
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
from scipy.io import savemat

from models import get_model
from models.KernelGAN import KernelGAN, Learner
from datasets.hr_lr_kernel_from_BioSR import HrLrKernelFromBioSR  # do not delete this line
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
    pred_kernel = gan.finish()
    return pred_kernel  # sum up to 1.0


def main():
    """please make sure that the pwd is .../PsfPred rather than .../PsfPred/codes/trains"""
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/train_something.yaml')
    args = prog.parse_args()

    # start train
    opt = read_yaml(args.opt)
    kernel_psnrs = []
    sr_psnrs = []
    mat_data = {'KernelGAN_pred_kernels': [], 'names': []}
    if not os.path.exists(opt['output_dir_path']):
        os.mkdir(opt['output_dir_path'])
        
    if opt['preload_data'] is None:
        print(f'load test data from {opt["input_image_root"]}')
        for file_name in os.listdir(opt['input_image_root']):
            opt['input_image_path'] = os.path.join(opt['input_image_root'], file_name)
            opt['img_name'] = os.path.splitext(opt['input_image_path'])[0]
            pred_kernel = train(opt)
            savemat(os.path.join(opt['output_dir_path'], '%s_kernel.mat' % os.path.basename(opt['img_name'])),
                    {'kernel': pred_kernel})        
    else:
        testset = pickle_load(opt['preload_data']).dataset
        print(f'load test data from {opt["preload_data"]}')
        F_model = get_model(opt['SFTMD_model']) if opt['do_SFTMD'] else None
        try:
            for i in range(len(testset)):
                data = testset[i]
                opt['input_image_path'] = data['name'] + '.png'
                opt['img_name'] = data['name']
                opt['input_image'] = torch.permute(data['lr'], (1, 2, 0)).contiguous().cpu().clone().numpy()
                pred_kernel = train(opt)
                mat_data['KernelGAN_pred_kernels'].append(pred_kernel)
                mat_data['names'].append(data['name'][0])
                pred_kernel = torch.from_numpy(pred_kernel)
                if opt['do_SFTMD']:
                    F_model.feed_data({'hr': data['hr'].unsqueeze(0), 'lr': data['lr'].unsqueeze(0),
                                       'kernel': pred_kernel.unsqueeze(0)})
                    F_model.test()
                gt_kernel = data['kernel']
                kernel_psnr = calculate_PSNR(pred_kernel, data['kernel'], max_val='auto')
                kernel_psnrs.append(kernel_psnr)
                sr = F_model.sr.squeeze(0).cpu() if opt['do_SFTMD'] else torch.rand(data['hr'].shape)
                sr_psnr = calculate_PSNR(data['hr'], F_model.sr.squeeze(0).cpu(), max_val=1.0) if opt['do_SFTMD'] else float('nan')
                sr_psnrs.append(sr_psnr)
                result = torch.cat([normalization(nearest_itpl(data['lr'], data['hr'].shape[-2:])),
                                    normalization(sr), normalization(data['hr'])], dim=-1)
                show_size = (data['hr'].shape[-2] // 4, data['hr'].shape[-1] // 4)
                pred_kernel = nearest_itpl(pred_kernel, show_size)
                gt_kernel = nearest_itpl(gt_kernel, show_size)
                result = overlap(normalization(gt_kernel), result, (0, 0))
                result = overlap(normalization(pred_kernel), result, (gt_kernel.shape[-2], 0))
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(data['hr'].shape[-2] // 25, 16)
                draw_text_on_image(result, f'PSNR {sr_psnr:5.2f}',
                                   (data['hr'].shape[-1], 0), font_size, 65535)
                draw_text_on_image(result, f'Kernel PSNR {kernel_psnr:5.2f}',
                                   (0, data['hr'].shape[-2] - 2 * font_size), font_size, 65535)
                draw_text_on_image(result, data['name'],
                                   (0, data['hr'].shape[-2] - font_size), font_size, 65535)
                result.save(os.path.join(opt['output_dir_path'], data['name'] + '.png'))
                print(f'psnr: kernel={kernel_psnr:5.2f}, sr={sr_psnr:5.2f}')
                print(f'current avg psnr: kernel={np.mean(kernel_psnrs):5.2f}, sr={np.mean(sr_psnrs):5.2f}\n')
            print(f'\n\navg psnr: kernel={np.mean(kernel_psnrs):5.2f}, sr={np.mean(sr_psnrs):5.2f}')
            savemat(os.path.join(opt['output_dir_path'], 'pred_kernels.mat'), mat_data)
        except:
            print(f'\n\navg psnr: kernel={np.mean(kernel_psnrs):5.2f}, sr={np.mean(sr_psnrs):5.2f}')
            savemat(os.path.join(opt['output_dir_path'], 'pred_kernels.mat'), mat_data)


if __name__ == '__main__':
    main()
