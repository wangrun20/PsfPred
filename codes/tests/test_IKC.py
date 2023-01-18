import argparse
import os
import torch
from torchvision import transforms
from tqdm import tqdm
from models import get_model
from datasets import get_dataloader
from utils.universal_util import read_yaml, calculate_PSNR, normalization, PCA_Decoder, nearest_itpl, overlap, \
    draw_text_on_image


def test(opt):
    # pass parameter
    correct_step = opt['testing']['correct_step']
    show_kernel = opt['testing']['show_kernel']
    show_kernel_code_psnr = opt['testing']['show_kernel_code_psnr']
    save_dir = opt['testing']['save_dir']

    # mkdir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up data loader
    test_loader = get_dataloader(opt['test_data'])

    # set up model
    F_model = get_model(opt['F_model'])
    P_model = get_model(opt['P_model'])
    C_model = get_model(opt['C_model'])

    pca_decoder = PCA_Decoder(weight=F_model.pca_encoder.weight, mean=F_model.pca_encoder.mean)

    # start testing
    with tqdm(desc=f'Testing', total=len(test_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for data in test_loader:
                P_model.feed_data(data)
                P_model.test()
                kernel_code_of_sr = P_model.pred_kernel_code.detach().cpu()
                gt_kernel_code = F_model.pca_encoder(data['kernel'].to(F_model.device))
                for i in range(correct_step):
                    F_model.feed_data({'hr': data['hr'],
                                       'lr': data['lr'],
                                       'kernel_code': kernel_code_of_sr})
                    F_model.test()
                    sr = F_model.sr.detach().cpu()
                    C_model.feed_data({'sr': sr,
                                       'kernel_code_of_sr': kernel_code_of_sr,
                                       'gt_kernel_code': gt_kernel_code})
                    C_model.test()
                    if i <= correct_step - 2:
                        kernel_code_of_sr = C_model.pred_kernel_code.detach().cpu()
                    else:
                        kernel_code_of_sr = kernel_code_of_sr.to(F_model.device)
                psnr = calculate_PSNR(F_model.hr, F_model.sr, max_val=1.0)
                # LR->SR->HR
                result = torch.cat([normalization(F_model.lr), normalization(F_model.sr),
                                    normalization(F_model.hr)], dim=-1).squeeze(0).squeeze(0)
                if show_kernel:
                    show_size = (F_model.hr.shape[-2] // 4, F_model.hr.shape[-1] // 4)
                    pred_kernel = pca_decoder(kernel_code_of_sr).squeeze(0)
                    gt_kernel = data['kernel'].to(F_model.device).squeeze(0)
                    kernel_psnr = calculate_PSNR(pred_kernel, gt_kernel, max_val='auto')
                    pred_kernel = nearest_itpl(pred_kernel, show_size)
                    gt_kernel = nearest_itpl(gt_kernel, show_size)
                    result = overlap(normalization(gt_kernel), result, (0, 0))
                    result = overlap(normalization(pred_kernel), result, (gt_kernel.shape[-2], 0))
                if show_kernel_code_psnr:
                    kernel_code_psnr = calculate_PSNR(kernel_code_of_sr, gt_kernel_code, max_val='auto')
                result = transforms.ToPILImage()((result * 65535).to(torch.int32))
                font_size = max(F_model.hr.shape[-2] // 25, 16)
                draw_text_on_image(result, data['name'][0], (result.width // 3 * 2, 0), font_size, 65535)
                draw_text_on_image(result, f'PSNR {psnr:5.2f}', (result.width // 3, 0), font_size, 65535)
                if show_kernel:
                    draw_text_on_image(result, f'Kernel PSNR {kernel_psnr:5.2f}',
                                       (0, F_model.hr.shape[-2] - 2 * font_size), font_size, 65535)
                if show_kernel_code_psnr:
                    draw_text_on_image(result, f'Code PSNR {kernel_code_psnr:5.2f}',
                                       (0, F_model.hr.shape[-2] - font_size), font_size, 65535)
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
