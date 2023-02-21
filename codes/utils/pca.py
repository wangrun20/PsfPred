import torch
from tqdm import tqdm

from zernike_psf import ZernikePSFGenerator
from utils.universal_util import get_phaseZ, pickle_dump, PCA_Encoder, PCA_Decoder, calculate_PSNR, normalization


def PCA(x, h=2):
    """
    :param x: (batch_size, num_feature)
    :param h:
    :return: (num_feature, h)
    """
    x_mean = torch.mean(x, dim=0, keepdim=True)
    x = x - x_mean
    U, S, V = torch.svd(torch.t(x))
    return U[:, :h]  # PCA matrix


def main():
    opt = {'kernel_size': 33,
           'NA': 1.35,
           'Lambda': 0.525,
           'RefractiveIndex': 1.33,
           'SigmaX': 2.0,
           'SigmaY': 2.0,
           'Pixelsize': 0.0313,
           'nMed': 1.33,
           'phaseZ': {'idx_start': 4,
                      'num_idx': 15,
                      'mode': 'gaussian',
                      'std': 0.125,
                      'bound': 1.0},
           'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
    norm = True
    psf_gen = ZernikePSFGenerator(opt)
    sample = []
    other = []
    total = 300000
    batch = 1000
    with tqdm(desc=f'generating...', total=total, unit='psf') as pbar:
        for _ in range(total // batch):
            s = psf_gen.generate_PSF(phaseZ=get_phaseZ(opt['phaseZ'], batch_size=batch, device=opt['device']))
            o = psf_gen.generate_PSF(phaseZ=get_phaseZ(opt['phaseZ'], batch_size=batch, device=opt['device']))
            if norm:
                s, o = normalization(s, batch=batch > 1), normalization(o, batch=batch > 1)
            sample.append(s)
            other.append(o)
            pbar.update(batch)
    sample = torch.cat(sample, dim=0)
    other = torch.cat(other, dim=0)
    flat = sample.view(sample.shape[0], -1)
    pca_mean = torch.mean(flat, dim=0, keepdim=True)
    pickle_dump(pca_mean.float().cpu(), './pca_mean.pth')
    for h in (10, 65, 92, 112):
        pca_matrix = PCA(flat, h=h)
        # h的值通过试探来确定，一般设定主成分贡献占比大于99%
        # (2023/1/11) 实验发现在当前配置下，h=65时占比99%(PSNR≈68)，h=92时占比99.9%(PSNR≈88)，h=112时占比99.99%(PSNR≈100)
        pickle_dump(pca_matrix.float().cpu(), f'./pca_matrix{h}.pth')
        zip_unzip = PCA_Decoder(pca_matrix, pca_mean)(PCA_Encoder(pca_matrix, pca_mean)(other))
        print(f'h={h}, PSNR={calculate_PSNR(other, zip_unzip, max_val="auto")}')


if __name__ == '__main__':
    main()
