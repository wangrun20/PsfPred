import os
import math
import torch
import pickle
import torch.nn.functional as F
from torchvision import transforms
from ruamel import yaml


def save_yaml(opt, yaml_path):
    f = open(yaml_path, 'w', encoding='utf-8')
    yaml.dump(opt, f, Dumper=yaml.RoundTripDumper)
    f.close()


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    data = yaml.load(f.read(), Loader=yaml.Loader)
    f.close()
    return data


def add_poisson_gaussian_noise(img, level=1000.0):
    if torch.max(img) == 0.0:
        poisson = torch.poisson(torch.zeros(*img.shape)).to(img.device)
    else:
        poisson = torch.poisson(img / torch.max(img) * level).to(img.device)
    gaussian = torch.normal(mean=torch.ones(*img.shape) * 100.0, std=torch.ones(*img.shape) * 4.5).to(img.device)
    img_noised = poisson + gaussian
    assert torch.max(img_noised) - torch.min(img_noised) != 0.0
    img_noised = (img_noised - torch.min(img_noised)) / (torch.max(img_noised) - torch.min(img_noised))
    if torch.max(img) != 0.0:
        img_noised = img_noised * (torch.max(img) - torch.min(img)) + torch.min(img)
    else:
        # raise RuntimeWarning('occur purely dark img')
        print('occur purely dark img')
    return img_noised


def random_rotate_crop_flip(img, new_size, fill):
    transformer = transforms.Compose([
        # bicubic may result in negative value of some pixels
        transforms.RandomRotation(degrees=(-180, +180), interpolation=transforms.InterpolationMode.BILINEAR, fill=fill),
        transforms.RandomCrop(size=new_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    return transformer(img)


def get_the_latest_file(root, postfix='_net.pth'):
    """
    target files should be named as '1234567_net.pth'
    """
    all_files = [file for file in os.listdir(root) if (file.endswith(postfix) and file.replace(postfix, '').isdigit())]
    assert len(all_files) > 0, 'empty directory'
    latest_file = all_files[0]
    get_step = lambda x: int(x.replace(postfix, ''))
    for certain_file in all_files:
        certain_step = get_step(certain_file)
        if certain_step > get_step(latest_file):
            latest_file = certain_file
    return {'path': os.path.join(root, latest_file),
            'step': get_step(latest_file)}


def remove_excess(root, keep, postfix='_model.pth'):
    all_files = [file for file in os.listdir(root) if file.endswith(postfix)]
    num_removed = 0
    for file in all_files:
        is_redundant = True
        for s in keep:
            if s in file:
                is_redundant = False
        if is_redundant:
            os.remove(os.path.join(root, file))
            num_removed += 1
    return num_removed


def calculate_PSNR(img1, img2, border=0, max_val=None):
    """
    input image shape should be torch.Tensor(..., H, W)
    border mean how many pixels of the edge will be abandoned. default: 0
    """
    assert len(img1.shape) >= 2 and len(img2.shape) >= 2, 'Input images must be in the shape of (..., H, W).'
    assert img1.shape == img2.shape, f'input images should have the same dimensions, ' \
                                     f'but got {img1.shape} and {img2.shape}'
    if max_val == 'auto':
        max_val = max(torch.max(img1).item(), torch.max(img2).item())
    H, W = img1.shape[-2:]
    img1 = img1[..., border:H - border, border:W - border].type(torch.float32)
    img2 = img2[..., border:H - border, border:W - border].type(torch.float32)
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 0.0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def normalization(tensor, v_min=0.0, v_max=1.0):
    if torch.max(tensor) - torch.min(tensor) == 0.0:
        return torch.clamp(tensor, max=v_max, min=v_min)
    return ((tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))) * (v_max - v_min) + v_min


def get_phaseZ(opt=None, batch_size=1, device=torch.device('cpu')):
    """
    opt: default = {'idx_start': 4, 'num_idx': 11, 'mode': 'gaussian', 'std': 0.125, 'bound': 1.0}
    """
    if opt is None:
        opt = {'idx_start': 4, 'num_idx': 11, 'mode': 'gaussian', 'std': 0.125, 'bound': 1.0}
    phaseZ = torch.zeros(size=(batch_size, 25))
    if opt['mode'] == 'gaussian':
        phaseZ[:, opt['idx_start']:opt['idx_start'] + opt['num_idx']] = torch.normal(mean=0.0, std=opt['std'],
                                                                                     size=(batch_size, opt['num_idx']))
        phaseZ = torch.clamp(phaseZ, min=-opt['bound'], max=opt['bound'])
    elif opt['mode'] == 'uniform':
        phaseZ[:, opt['idx_start']:opt['idx_start'] + opt['num_idx']] = torch.rand(size=(batch_size, opt['num_idx'])) \
                                                                        * 2.0 * opt['bound'] - opt['bound']
    else:
        raise NotImplementedError
    return phaseZ.to(device)


def pickle_dump(obj, file_path):
    with open(file_path, "xb") as f:
        pickle.dump(obj, f)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def rectangular_closure(x):
    """返回矩形闭包的上下左右边界"""
    assert x.dtype == torch.bool
    assert len(x.shape) == 2
    a_h = torch.argwhere(torch.sum(x, dim=1) >= 1)
    a_w = torch.argwhere(torch.sum(x, dim=0) >= 1)
    return a_h[0].item(), a_h[-1].item(), a_w[0].item(), a_w[-1].item()


def scale_and_translation(x, v_max, v_min):
    x = normalization(x)
    return x * (v_max - v_min) + v_min


def complex_to_reals(x, mode='ri'):
    """
    element of x should be complex number
    mode = 'ri' | 'ap'
    """
    if mode == 'ri':
        return x.real, x.imag
    elif mode == 'ap':
        a = torch.angle(x)
        # a = a.detach().cpu().numpy()
        # a = np.unwrap(np.unwrap(a, axis=-1), axis=-2)
        # a = torch.from_numpy(a).to(x.device)
        return torch.abs(x), a
    else:
        raise NotImplementedError


def one_plus_log(x, base='e'):
    if base == 'e':
        return torch.sign(x) * torch.log(1.0 + torch.abs(x))
    elif base == '10':
        return torch.sign(x) * torch.log10(1.0 + torch.abs(x))
    else:
        raise NotImplementedError


def save_gray_img(x, path, norm=True):
    assert len(x.shape) == 2
    if norm:
        x = normalization(x)
    x = (x * 65535.0).to(torch.int32)
    transforms.ToPILImage()(x).save(path)


class PCA_Encoder(object):
    def __init__(self, weight, mean):
        self.weight = weight  # l**2 x h
        self.mean = mean  # 1 x l**2
        self.size = self.weight.size()

    def __call__(self, batch_kernel):
        """
        :param batch_kernel: shape (B, l, l)
        :return: shape (B, h)
        """
        B, H, W = batch_kernel.size()  # (B, l, l)
        return torch.bmm(batch_kernel.view((B, 1, H * W)) - self.mean,
                         self.weight.expand((B, ) + self.size)).view((B, -1))


class PCA_Decoder(object):
    def __init__(self, weight, mean):
        self.weight = weight  # l**2 x h
        self.mean = mean  # 1 x l**2
        self.l = int(self.weight.shape[0] ** 0.5)
        self.size = weight.T.size()
        assert self.l * self.l == self.weight.shape[0]

    def __call__(self, batch_kernel_code):
        """
        :param batch_kernel_code: shape (B, h)
        :return: shape (B, l, l)
        """
        B, h = batch_kernel_code.shape
        return (torch.bmm(batch_kernel_code.view((B, 1, h)), self.weight.T.expand((B, ) + self.size)) + self.mean).view(
            (B, self.l, self.l))


def nearest_itpl(x, size):
    """nearest interpolation"""
    assert len(size) == 2
    if len(x.shape) == 4:
        return F.interpolate(x, size, mode='nearest')
    if len(x.shape) == 3:
        return F.interpolate(x.unsqueeze(0), size, mode='nearest').squeeze(0)
    if len(x.shape) == 2:
        return F.interpolate(x.unsqueeze(0).unsqueeze(0), size, mode='nearest').squeeze(0).squeeze(0)


def overlap(x, y, pos):
    """put x on y, and x[..., 0, 0] at pos"""
    y = y.clone().detach()
    y[..., pos[-2]:pos[-2] + x.shape[-2], pos[-1]:pos[-1] + x.shape[-1]] = x
    return y


def main():
    opt = read_yaml('../options/train_FFTRCANResUNet.yaml')
    print(opt)
    pass


if __name__ == '__main__':
    main()
