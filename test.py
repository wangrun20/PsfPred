import argparse
import torch
import torch.nn.functional as F

from torchvision import transforms
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from PIL import ImageFont, ImageDraw
from tqdm import tqdm
from general_utils import calculate_PSNR, normalization, read_yaml, complex_to_reals, one_plus_log, add_poisson_gaussian_noise
from dataset import get_dataloader
from models import get_model, restore_checkpoint, \
    FFTResUNet, FFTRCANResUNet, FFTOnlyRCANResUNet, FreqDomainRCANResUNet, \
    MANet_s1, psnr_heat_map


def test_on_given_net_data(net, data_loader, save_path, loss_func=torch.nn.MSELoss()):
    assert data_loader.batch_size == 1, 'batch_size of test data loader should be 1'
    net.eval()
    net_device = next(net.parameters()).device
    loss = 0.0
    results = []
    psnrs = []
    normed_psnrs = []
    if type(net) == torch.nn.DataParallel:
        net_type = type(net.module)
    else:
        net_type = type(net)
    with tqdm(desc=f'testing', total=len(data_loader.dataset), unit='img') as pbar:
        with torch.no_grad():
            for batch in data_loader:
                if net_type in (FFTResUNet, FFTRCANResUNet, FFTOnlyRCANResUNet, MANet_s1):
                    lr = batch['lr'].to(net_device)
                    assert lr.shape[-2] % 2 == 0
                    k_pred = net(lr)
                    k_true = (batch['kernel'] if len(batch['kernel'].shape) >= 3 else torch.zeros(*k_pred.shape)).to(
                        net_device)
                    if net_type == MANet_s1:
                        heat_map = psnr_heat_map(k_true.squeeze(0).squeeze(0), k_pred.squeeze(0).view(lr.shape[-2], lr.shape[-1], k_true.shape[-2], k_true.shape[-1]), is_norm=True)
                        max_heat = torch.max(heat_map).item()
                        min_heat = torch.min(heat_map).item()
                        heat_map = normalization(heat_map.float())
                        k_pred = torch.mean(k_pred, dim=1, keepdim=True)
                    loss += loss_func(k_true, k_pred).item()
                    psnr = calculate_PSNR(k_pred.detach(), k_true.detach(),
                                          max_val=max(torch.max(k_true).item(), torch.max(k_pred).item()))
                    k_pred = normalization(F.interpolate(k_pred,
                                                         size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                         mode='nearest')).squeeze(0).squeeze(0)
                    k_true = normalization(F.interpolate(k_true,
                                                         size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                         mode='nearest')).squeeze(0).squeeze(0)
                    normed_psnr = calculate_PSNR(k_pred.detach(), k_true.detach(),
                                                 max_val=max(torch.max(k_true).item(), torch.max(k_pred).item()))
                    lr = normalization(lr).squeeze(0).squeeze(0)
                    if net_type != MANet_s1:
                        result = torch.cat([lr, torch.cat([k_true, k_pred], dim=-2)], dim=-1)
                    else:
                        result = torch.cat([lr, heat_map, torch.cat([k_true, k_pred], dim=-2)], dim=-1)
                elif net_type in (FreqDomainRCANResUNet,):
                    lr = batch['hr'].to(net_device)
                    lr = add_poisson_gaussian_noise(lr, 1000)
                    assert lr.shape[-2] % 2 == 0
                    psf_gen = data_loader.dataset.psf_gen
                    x1, x2 = net.get_input(lr)
                    pupil_phase_pred = net(lr)
                    pupil_phase_true = psf_gen.mask_pupil_phase(
                        psf_gen.phaseZ_to_pupil_phase(batch['phaseZ'].to(psf_gen.device).squeeze(1))).to(
                        net_device).unsqueeze(0)
                    k_pred = psf_gen.pupil_phase_to_PSF(
                        psf_gen.pad_pupil_phase(pupil_phase_pred.squeeze(1)).to(psf_gen.device)).to(
                        net_device).unsqueeze(1)

                    k_true = (batch['kernel'] if batch['kernel'] is not None else torch.zeros(*k_pred.shape)).to(
                        net_device)
                    loss += loss_func(k_true, k_pred).item()
                    psnr = calculate_PSNR(k_pred.detach(), k_true.detach(),
                                          max_val=max(torch.max(k_true).item(), torch.max(k_pred).item()))

                    lr_fft = torch.fft.fftshift(torch.fft.fft2(lr))
                    fft1, fft2 = complex_to_reals(lr_fft, mode='ap')
                    fft1 = one_plus_log(fft1)
                    fft2 = one_plus_log(fft2)
                    fft1 = normalization(fft1).squeeze(0).squeeze(0)
                    fft2 = normalization(fft2).squeeze(0).squeeze(0)
                    lr = normalization(lr).squeeze(0).squeeze(0)
                    x1 = normalization(F.interpolate(x1, size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                     mode='nearest')).squeeze(0).squeeze(0)
                    x2 = normalization(F.interpolate(x2, size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                     mode='nearest')).squeeze(0).squeeze(0)
                    k_pred = normalization(F.interpolate(k_pred, size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                         mode='nearest')).squeeze(0).squeeze(0)
                    k_true = normalization(F.interpolate(k_true, size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                         mode='nearest')).squeeze(0).squeeze(0)
                    normed_psnr = calculate_PSNR(k_pred.detach(), k_true.detach(),
                                                 max_val=max(torch.max(k_true).item(), torch.max(k_pred).item()))
                    pupil_phase_pred = (F.interpolate(pupil_phase_pred, size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                      mode='nearest')).squeeze(0).squeeze(0)
                    pupil_phase_true = (F.interpolate(pupil_phase_true, size=(lr.shape[-2] // 2, lr.shape[-2] // 2),
                                                      mode='nearest')).squeeze(0).squeeze(0)
                    pupil_phase_p_t = normalization(torch.cat([pupil_phase_true, pupil_phase_pred], dim=-2))
                    result = torch.cat([torch.cat([lr, fft1, fft2], dim=-1),
                                        torch.cat([x1, x2], dim=-2),
                                        pupil_phase_p_t,
                                        torch.cat([k_true, k_pred], dim=-2)], dim=-1)
                else:
                    raise NotImplementedError

                result = (result * 65535.0).to(torch.int32)
                result = transforms.ToPILImage()(result)
                draw_ks_lr = ImageDraw.Draw(result)
                my_font = ImageFont.truetype('/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf', size=16)
                draw_ks_lr.text((0, 0), batch['name'][0], font=my_font, fill=65535)
                if net_type == MANet_s1:
                    draw_ks_lr.text((lr.shape[-1], 0), f'{min_heat:5.2f}~{max_heat:5.2f}', font=my_font, fill=65535)
                draw_ks_lr.text((int(result.width - k_true.shape[-1]), 0),
                                f'PSNR {psnr:5.2f}' if psnr is not None else f'No PSNR', font=my_font, fill=65535)
                draw_ks_lr.text((int(result.width - k_true.shape[-1]), result.height // 2),
                                f'N_PSNR {normed_psnr:5.2f}' if psnr is not None else f'No PSNR', font=my_font, fill=65535)
                result = transforms.ToTensor()(result).squeeze(0)
                results.append(result.cpu())
                psnrs.append(psnr)
                normed_psnrs.append(normed_psnr)
                pbar.update(1)
    num_test = len(results)
    if num_test == 40:
        col, row = 8, 5
    elif num_test == 20:
        col, row = 4, 5
    elif num_test == 10:
        col, row = 2, 5
    else:
        col = int(num_test ** 0.5)
        row = num_test // col
        col += (1 if col * row < num_test else 0)
    results += [torch.zeros(size=result.shape, dtype=torch.int32) for _ in range(row * col - len(results))]
    results = torch.cat([torch.cat([results[i * col + j] for j in range(col)], dim=-1) for i in range(row)], dim=-2)
    if save_path is not None:
        transforms.ToPILImage()(results).save(save_path)
    return {"test_loss": loss / num_test,
            'test_avg_psnr': sum(psnrs) / len(psnrs),
            'test_avg_n_psnr': sum(normed_psnrs) / len(normed_psnrs)}


def run_test(opt_path: str):
    opt = read_yaml(opt_path)

    if opt['test_settings']['model_gpu_id'] is not None:
        device = torch.device('cuda', opt['test_settings']['model_gpu_id'])
    else:
        device = torch.device('cpu')

    paths_nets = []
    for net_opt in opt['model']:
        net = get_model(net_opt)
        if opt['test_settings']['is_data_parallel']:
            net = torch.nn.DataParallel(net).cuda()
        path = restore_checkpoint(net, net_opt['checkpoint_root'],
                                  mode=net_opt['mode'], postfix='_net.pth')['path']
        paths_nets.append({'path': path,
                           'net': net})

    test_loaders = [get_dataloader(x) for x in opt['test_dataset']]

    name_base = 'result'
    is_save_img = opt['test_settings']['is_save_img']

    for i, path_net in enumerate(paths_nets):
        for j, test_loader in enumerate(test_loaders):
            path = path_net['path']
            if opt['test_settings']['is_data_parallel']:
                net = path_net['net']
            else:
                net = path_net['net'].to(device)
            if type(net) == MANet_s1:
                test_loader.dataset.is_norm_lr = False
                test_loader.dataset.is_norm_k = False
            else:
                test_loader.dataset.is_norm_lr = True
                test_loader.dataset.is_norm_k = True
            print(f'restore checkpoint from {path}')
            test_result = test_on_given_net_data(net, test_loader, f'./{name_base}_m{j}n{i}.png' if is_save_img else None)
            print('avg loss = {}'.format(test_result['test_loss']))
            print('avg psnr = {}'.format(test_result['test_avg_psnr']))
            print('avg n_psnr = {}'.format(test_result['test_avg_n_psnr']))
            print()
        print()
        del net  # release (cuda) memory


def main():
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/test.yaml')
    args = prog.parse_args()

    run_test(args.opt)


if __name__ == '__main__':
    main()
