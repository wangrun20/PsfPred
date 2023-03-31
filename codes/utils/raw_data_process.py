import os
import shutil
import multipagetiff as mtif
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def add_1to9():
    root = r'F:\DAO_WR\20230219_COS7_MT_3XmEmerald_Enconsin\LowNA_GI-SIM'
    dirs = os.listdir(root)
    name = 'roi1_seq1_Low NA GI-SIM488_GreenCh.tif'
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, name)):
            s = mtif.read_stack(os.path.join(root, dir, name))
            s = np.array(s)
            for j in range(s.shape[0] // 9):
                img = s[0, :, :] * 0
                img = np.asarray(img, dtype=np.int32)
                for i in range(9):
                    img += s[i + 9 * j, :, :]
                assert np.min(img) >= 0.0
                if np.max(img) > 65535:
                    img = np.asarray(img, dtype=float)
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                    img *= 65535
                    img = np.asarray(img, dtype=np.int32)
                img = transforms.ToPILImage()(torch.from_numpy(img))
                img.save(os.path.join(root, dir, f'{9 * j + 1}-{9 * j + 9}.png'))
            print(f'{dir} done')


def extract_recons():
    root = r'F:\DAO_WR\20230221_SUM_ki_CCPs_Clathrin'
    dirs = os.listdir(root)
    name = 'roi1_seq1_TIRF-SIM488_GreenCh_SIrecon.tif'
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, name)):
            s = mtif.read_stack(os.path.join(root, dir, name))
            s = np.array(s)
            img = s[0, :, :]
            img = np.asarray(img, dtype=float)
            if np.min(img) < 0:
                img -= np.min(img)
            if np.max(img) > 65535:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
            img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(root, dir, 'recons.png'))


def copy_rename():
    root = r'F:\DAO_WR\20230219_COS7_MT_3XmEmerald_Enconsin\LowNA_GI-SIM'
    dirs = os.listdir(root)
    for dir in dirs:
        os.mkdir(os.path.join(r'C:\Mine\PsfPred\data\exp_timeline', dir))
        i = 0
        while True:
            if os.path.exists(os.path.join(root, dir, f'{i * 9 + 1}-{i * 9 + 9}.png')):
                shutil.copy(os.path.join(root, dir, f'{i * 9 + 1}-{i * 9 + 9}.png'),
                            os.path.join(r'C:\Mine\PsfPred\data\exp_timeline', dir, f'{i * 9 + 1}-{i * 9 + 9}.png'))
                i += 1
            else:
                break
        print(f'{dir} done')


def scan_pos(H, W, h, w, s=2):
    """
    scan big picture (H, W) with small picture (h, w)
    :param H:
    :param W:
    :param h:
    :param w:
    :param s:
    :return:
    """
    assert H > h and W > w
    assert h % s == 0 and w % s == 0
    y, x = 0, 0
    ys, xs = [], []
    positions = []
    while True:
        ys.append(y)
        if y + h > H:
            ys[-1] = H - h
            break
        y += (h // s)
    while True:
        xs.append(x)
        if x + w > W:
            xs[-1] = W - w
            break
        x += (w // s)
    for y in ys:
        for x in xs:
            positions.append((y, x))
    return positions


def crop(h=132, w=132):
    root = r'C:\Mine\PsfPred\data\exp-data'
    to = r'C:\Mine\PsfPred\data\exp-crop'
    dirs = os.listdir(root)
    for dir in dirs:
        names = os.listdir(os.path.join(root, dir))
        for name in names:
            img = Image.open(os.path.join(root, dir, name))
            H, W = img.height, img.width
            postions = scan_pos(H, W, h, w)
            for i, pos in enumerate(postions):
                cropped_img = img.crop(box=(pos[1], pos[0], pos[1] + w, pos[0] + h))
                cropped_img.save(os.path.join(to, dir, name.replace('.png', f'({str(i + 1).rjust(2, "0")}).png')))


if __name__ == '__main__':
    add_1to9()
    # extract_recons()
    copy_rename()
    # crop()
