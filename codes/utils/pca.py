import torch
from zernike_psf import ZernikePSFGenerator
from utils.universal_util import get_phaseZ, read_yaml, pickle_dump, PCA_Encoder, PCA_Decoder, calculate_PSNR


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
    opt = read_yaml('./options/train_SFTMD.yaml')['train_data']['psf_settings']
    opt['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    psf_gen = ZernikePSFGenerator(opt)
    sample = []
    other = []
    for _ in range(3000):
        sample.append(psf_gen.generate_PSF(phaseZ=get_phaseZ(opt['phaseZ'], batch_size=100, device=opt['device'])))
        other.append(psf_gen.generate_PSF(phaseZ=get_phaseZ(opt['phaseZ'], batch_size=100, device=opt['device'])))
    sample = torch.cat(sample, dim=0)
    other = torch.cat(other, dim=0)
    flat = sample.view(sample.shape[0], -1)
    pca_mean = torch.mean(flat, dim=0, keepdim=True)
    pickle_dump(pca_mean.float().cpu(), './pca_mean.pth')
    for h in (65, 92, 112):
        pca_matrix = PCA(flat, h=h)
        # h的值通过试探来确定，一般设定主成分贡献占比大于99%
        # (2023/1/11) 实验发现在当前配置下，h=65时占比99%(PSNR≈68)，h=92时占比99.9%(PSNR≈88)，h=112时占比99.99%(PSNR≈100)
        pickle_dump(pca_matrix.float().cpu(), f'./pca_matrix{h}.pth')
        zip_unzip = PCA_Decoder(pca_matrix, pca_mean)(PCA_Encoder(pca_matrix, pca_mean)(other))
        print(f'h={h}, PSNR={calculate_PSNR(other, zip_unzip, max_val="auto")}')


if __name__ == '__main__':
    main()
