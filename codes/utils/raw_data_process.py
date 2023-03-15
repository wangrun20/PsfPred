import os
import shutil
import multipagetiff as mtif
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def main():
    root = r'F:\DAO_WR\20230219_COS7_MT_3XmEmerald_Enconsin\HighNA_GI-SIM'
    dirs = os.listdir(root)
    name = 'roi1_seq1_High NA GI-SIM488_GreenCh.tif'
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, name)):
            s = mtif.read_stack(os.path.join(root, dir, name))
            s = np.array(s)
            img = s[0, :, :] * 0
            img = np.asarray(img, dtype=np.int32)
            for i in range(9):
                img += s[i, :, :]
            if np.max(img) > 65535:
                img = np.asarray(img, dtype=float)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img *= 65535
                img = np.asarray(img, dtype=np.int32)
            img = transforms.ToPILImage()(torch.from_numpy(img))
            img.save(os.path.join(root, dir, '1-9.png'))


def copy_rename():
    root = r'F:\DAO_WR\20230219_COS7_MT_3XmEmerald_Enconsin\HighNA_GI-SIM'
    dirs = os.listdir(root)
    for dir in dirs:
        if os.path.exists(os.path.join(root, dir, '1-9.png')):
            shutil.copy(os.path.join(root, dir, '1-9.png'), os.path.join(r'C:\Mine\PsfPred\data\exp-raw', f'{dir}.png'))


def crop():
    root = r'C:\Mine\PsfPred\data\exp-raw'
    to = r'C:\Mine\PsfPred\data\exp-264'
    dirs = os.listdir(root)
    for dir in dirs:
        names = os.listdir(os.path.join(root, dir))
        for name in names:
            img = Image.open(os.path.join(root, dir, name))
            H, W = img.height, img.width
            img = img.crop(box=(W // 2 - 66, H // 2 - 66, W // 2 + 66, H // 2 + 66))
            img.save(os.path.join(to, dir, name))


if __name__ == '__main__':
    # main()
    # copy_rename()
    crop()
