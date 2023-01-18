import torch

from models.base_model import BaseModel
from utils import pickle_load, PCA_Encoder


class F_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.hr = torch.rand(size=(16, 1, 224, 224))
        self.lr = torch.rand(size=(16, 1, 224, 224))
        self.sr = torch.rand(size=(16, 1, 224, 224))
        self.kernel = torch.rand(size=(16, 33, 33))  # (B, l, l)
        self.pca_encoder = PCA_Encoder(weight=pickle_load(opt['pca_matrix']).to(self.device),
                                       mean=pickle_load(opt['pca_mean']).to(self.device))
        self.kernel_code = torch.rand(size=(16, 25))  # (B, h)

    def feed_data(self, data):
        self.hr = data['hr'].to(self.device)
        self.lr = data['lr'].to(self.device)
        if 'kernel' in data.keys() and 'kernel_code' not in data.keys():
            self.kernel = data['kernel'].to(self.device)
            self.kernel_code = self.pca_encoder(self.kernel)
        else:
            self.kernel = None
            self.kernel_code = data['kernel_code'].to(self.device)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.sr = self.network(self.lr, self.kernel_code)
            self.loss = self.loss_function(self.hr, self.sr) if self.loss_function is not None else None
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.sr = self.network(self.lr, self.kernel_code)
            self.loss = self.loss_function(self.hr, self.sr)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError


class P_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.lr = torch.rand(size=(16, 1, 224, 224))
        self.gt_kernel = torch.rand(size=(16, 33, 33))
        self.pca_encoder = PCA_Encoder(weight=pickle_load(opt['pca_matrix']).to(self.device),
                                       mean=pickle_load(opt['pca_mean']).to(self.device))
        self.gt_kernel_code = torch.rand(size=(16, 10))
        self.pred_kernel_code = torch.rand(size=(16, 10))

    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)
        self.gt_kernel = data['kernel'].to(self.device)
        self.gt_kernel_code = self.pca_encoder(self.gt_kernel)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.pred_kernel_code = self.network(self.lr)
            self.loss = self.loss_function(self.gt_kernel_code, self.pred_kernel_code) if self.loss_function is not None else None
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.pred_kernel_code = self.network(self.lr)
            self.loss = self.loss_function(self.gt_kernel_code, self.pred_kernel_code)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError


class C_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.sr = torch.rand(size=(16, 1, 224, 224))
        self.kernel_code_of_sr = torch.rand(size=(16, 10))
        self.gt_kernel_code = torch.rand(size=(16, 10))
        self.pred_kernel_code = torch.rand(size=(16, 10))

    def feed_data(self, data):
        self.sr = data['sr'].to(self.device)
        self.kernel_code_of_sr = data['kernel_code_of_sr'].to(self.device)
        self.gt_kernel_code = data['gt_kernel_code'].to(self.device)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.pred_kernel_code = self.network(self.sr, self.kernel_code_of_sr)
            self.loss = self.loss_function(self.gt_kernel_code, self.pred_kernel_code) if self.loss_function is not None else None
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.pred_kernel_code = self.network(self.sr, self.kernel_code_of_sr)
            self.loss = self.loss_function(self.gt_kernel_code, self.pred_kernel_code)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError
