import argparse
import os
from tqdm import tqdm

from models.KernelGAN import KernelGAN, Learner
from datasets.KernelGAN_data_generator import DataGenerator
from utils.universal_util import read_yaml


def train(opt):
    gan = KernelGAN(opt)
    learner = Learner()
    data = DataGenerator(opt, gan)
    for iteration in tqdm(range(opt['max_iters']), ncols=60):
        [g_in, d_in] = data[iteration]
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    gan.finish()


def main():
    """please make sure that the pwd is .../PsfPred rather than .../PsfPred/codes/trains"""
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/train_something.yaml')
    args = prog.parse_args()

    # start train
    opt = read_yaml(args.opt)
    if not os.path.exists(opt['output_dir_path']):
        os.mkdir(opt['output_dir_path'])
    for file_name in os.listdir(opt['input_image_root']):
        opt['input_image_path'] = os.path.join(opt['input_image_root'], file_name)
        opt['img_name'] = os.path.splitext(opt['input_image_path'])[0]
        train(opt)


if __name__ == '__main__':
    main()
