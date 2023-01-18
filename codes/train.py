import argparse
import os
import shutil
import torch
import wandb
from tqdm import tqdm
from wandb.util import generate_id

from utils import read_yaml, remove_excess, pickle_dump, pickle_load
from dataset import get_dataloader
from models import get_model, restore_checkpoint, \
    FFTResUNet, FFTRCANResUNet, FFTOnlyRCANResUNet, FreqDomainRCANResUNet, \
    MANet_s1
from test import test_on_given_net_data


def train(opt_path: str):
    # read configuration
    opt = read_yaml(opt_path)

    # pass parameter
    project_name = opt['project_name']
    experiment_path = os.path.join('../experiments', opt['experiment_name'])
    train_settings = opt['train_settings']
    is_data_parallel = train_settings['is_data_parallel']
    model_gpu_id = train_settings['model_gpu_id']
    max_epoch = train_settings['max_epoch']
    batch_size = opt['train_dataset']['loader_settings']['batch_size']
    learning_rate = train_settings['learning_rate']
    imgs_per_val_and_save = train_settings['imgs_per_val_and_save']
    imgs_per_scheduler_step = train_settings['imgs_per_scheduler_step']
    resume = train_settings['resume']
    assert imgs_per_val_and_save % batch_size == 0, \
        'imgs_per_val_and_save should be divided by batch_size with no remainder'
    assert imgs_per_scheduler_step % batch_size == 0, \
        'imgs_per_scheduler_step should be divided by batch_size with no remainder'

    # set up data loader
    train_loader = get_dataloader(opt['train_dataset'])

    # set up model, optimizer, scheduler, loss function, step
    net = get_model(opt['model'])
    print(f'total parameters: {sum(x.numel() for x in net.parameters())}')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate if learning_rate is not None else 1e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)
    if train_settings['loss'] == 'L1':
        loss_func = torch.nn.L1Loss()
    elif train_settings['loss'] == 'L2':
        loss_func = torch.nn.MSELoss()
    else:
        raise NotImplementedError
    step = 1

    # restore from checkpoint optionally
    if resume == 'auto':
        is_restore = os.path.exists(experiment_path)
        is_pth = False
        if is_restore:
            for file in os.listdir(experiment_path):
                if '.pth' in file:
                    is_pth = True
                    break
    else:
        raise NotImplementedError
    if is_restore:
        # restore model, optimizer, scheduler
        if is_pth:
            path1_step = restore_checkpoint(net, experiment_path, mode='latest', postfix='_net.pth')
            path2_step = restore_checkpoint(optimizer, experiment_path, mode='latest', postfix='_opt.pth')
            path3_step = restore_checkpoint(scheduler, experiment_path, mode='latest', postfix='_scd.pth')
            # check steps
            assert path1_step['step'] == path2_step['step'] == path3_step['step'], \
                'checkpoints of net, optimizer and scheduler are not from the same step'
            step = path1_step['step']
            print('model restored from {path}, step is {step}'.format(**path1_step))
            print('optimizer restored from {path}, step is {step}'.format(**path2_step))
            print('scheduler restored from {path}, step is {step}'.format(**path3_step))
        # reset learning rate
        if learning_rate is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
        # move optimizer params to gpu
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(model_gpu_id)
        # try to restore test data loader
        if os.path.exists(os.path.join(experiment_path, './test_data_loader.pkl')):
            test_loader = pickle_load(os.path.join(experiment_path, './test_data_loader.pkl'))
            print('test_loader restored')
        else:
            test_loader = get_dataloader(opt['test_dataset'])
            pickle_dump(test_loader, os.path.join(experiment_path, './test_data_loader.pkl'))
            print('new test_data_loader created')
        # restore wandb_id
        if os.path.exists(os.path.join(experiment_path, 'wandb_id.txt')):
            with open(os.path.join(experiment_path, 'wandb_id.txt'), 'r') as f:
                wandb_id = f.read()
        else:
            wandb_id = generate_id()
            with open(os.path.join(experiment_path, 'wandb_id.txt'), 'x') as f:
                f.write(wandb_id)
    else:
        # check directory
        assert not (os.path.exists(experiment_path) and len(os.listdir(experiment_path)) != 0), \
            'experiment directory already exists and contains files, but choose not to restore checkpoint'
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)
        # back up configuration
        shutil.copyfile(opt_path, os.path.join(experiment_path, opt_path.replace('\\', '/').split('/')[-1]))
        # try to restore test data loader
        if os.path.exists(os.path.join(experiment_path, './test_data_loader.pkl')):
            test_loader = pickle_load(os.path.join(experiment_path, './test_data_loader.pkl'))
            print('test_loader restored')
        else:
            test_loader = get_dataloader(opt['test_dataset'])
            pickle_dump(test_loader, os.path.join(experiment_path, './test_data_loader.pkl'))
            print('new test_data_loader created')
        # create and back up wandb_id
        wandb_id = generate_id()
        with open(os.path.join(experiment_path, 'wandb_id.txt'), 'x') as f:
            f.write(wandb_id)

    # move model to gpu
    net_type = type(net)
    if is_data_parallel:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda(model_gpu_id)

    # set up wandb
    print(f'wandb_id is {wandb_id}')
    wandb.init(project=project_name,
               name=opt['experiment_name'],
               config={"net": opt['model']['network'],
                       "epochs": max_epoch,
                       "batch_size": batch_size,
                       "learning_rate": learning_rate,
                       'img_signal': (opt['train_dataset']['img_signal'][0], opt['train_dataset']['img_signal'][-1]),
                       'psf_settings': opt['train_dataset']['psf_settings']},
               resume='allow',
               id=wandb_id)

    # initialize best loss
    if is_restore:
        try:
            best_file = [file for file in os.listdir(experiment_path) if (file.endswith('_net.pth') and 'best' in file)]
            if len(best_file) == 0:
                raise FileNotFoundError
            if len(best_file) > 1:
                raise RuntimeError('there are more than one best checkpoints')
            best_file = best_file[0]
            best_loss = float(best_file.split('_')[1])
        except FileNotFoundError:
            best_loss = float('inf')
    else:
        best_loss = float('inf')

    # start training
    for epoch in range(1, max_epoch + 1):
        cumulative_loss = 0.0
        with tqdm(desc=f'Epoch {epoch}/{max_epoch}', total=len(train_loader.dataset), unit='img') as pbar:
            for batch in train_loader:
                # feed data and run backpropagation
                net.train()
                if net_type in (FFTResUNet, FFTRCANResUNet, FFTOnlyRCANResUNet):
                    lr = batch['lr'].cuda(model_gpu_id)
                    kernel_true = batch['kernel'].cuda(model_gpu_id)
                    kernel_pred = net(lr)
                    loss = loss_func(kernel_pred, kernel_true)
                elif net_type in (FreqDomainRCANResUNet,):
                    lr = batch['lr'].cuda(model_gpu_id)
                    psf_gen = train_loader.dataset.psf_gen
                    assert psf_gen.mask_l == (net.module.mask_l if type(net) == torch.nn.DataParallel else net.mask_l)
                    pupil_phase_pred = net(lr).squeeze(1)
                    pupil_phase_true = psf_gen.phaseZ_to_pupil_phase(batch['phaseZ'].to(psf_gen.device).squeeze(1))
                    pupil_phase_true = psf_gen.mask_pupil_phase(pupil_phase_true).cuda(model_gpu_id)
                    loss = loss_func(pupil_phase_true, pupil_phase_pred)
                elif net_type in (MANet_s1,):
                    lr = batch['lr'].cuda(model_gpu_id)
                    kernel_true = batch['kernel'].cuda(model_gpu_id)
                    kernel_pred = net(lr)
                    kernel_true = kernel_true.expand(-1, kernel_pred.shape[1], -1, -1)
                    loss = loss_func(kernel_pred, kernel_true)
                else:
                    raise NotImplementedError
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record information
                cumulative_loss += loss.item()

                # log wandb
                wandb.log({
                    "tr_loss": loss.item(),
                    'step': step,
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

                # validate at the beginning of training
                if step == 1:
                    test_on_given_net_data(net, test_loader, os.path.join(experiment_path, f'{step}_val.png'), loss_func)

                # validate and save model at intervals
                if step % (imgs_per_val_and_save // batch_size) == 0:
                    test_info = test_on_given_net_data(net, test_loader, os.path.join(experiment_path, f'{step}_val.png'), loss_func)
                    test_info['step'] = step
                    wandb.log(test_info)
                    if test_info['test_loss'] < best_loss:
                        best_loss = test_info['test_loss']
                        for postfix in ['_net.pth', '_opt.pth', '_scd.pth']:
                            remove_excess(experiment_path, (), postfix)
                        torch.save(net.module.state_dict() if is_data_parallel else net.state_dict(),
                                   os.path.join(experiment_path, f'best_{best_loss:.2e}_{step}_net.pth'))
                    torch.save(net.module.state_dict() if is_data_parallel else net.state_dict(),
                               os.path.join(experiment_path, f'{step}_net.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(experiment_path, f'{step}_opt.pth'))
                    torch.save(scheduler.state_dict(), os.path.join(experiment_path, f'{step}_scd.pth'))
                    for postfix in ['_net.pth', '_opt.pth', '_scd.pth']:
                        remove_excess(experiment_path, (str(step), 'best'), postfix)

                # step scheduler at intervals
                if step % (imgs_per_scheduler_step // batch_size) == 0:
                    scheduler.step(cumulative_loss)
                    cumulative_loss = 0.0

                # update info
                step += 1
                pbar.update(lr.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item(),
                                    'lr': optimizer.param_groups[0]['lr']})


def main():
    # set up cmd
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str, default='./options/train_MANet.yaml')
    args = prog.parse_args()

    # run this command to allow data exchange between processes
    torch.multiprocessing.set_start_method('spawn', force=True)

    # start train
    train(args.opt)


if __name__ == '__main__':
    main()
