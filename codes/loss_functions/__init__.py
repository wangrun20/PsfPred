import torch.nn as nn


def get_loss_function(opt):
    match opt['name']:
        case None:
            return None
        case 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        case 'L1':
            return nn.L1Loss()
        case 'MSE':
            return nn.MSELoss()
        case _:
            raise NotImplementedError
