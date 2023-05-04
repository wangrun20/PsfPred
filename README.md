# Complex Optical Aberration Estimation via Spatial-Frequency Encoding Network Enables Sensitive Digital Adaptive Optics and Super-resolution Imaging

## Repository Usage

所有Python代码都在`./codes`中。
1. `./codes/networks`仅包含神经网络结构的定义，
实现了IKC、KernelGAN、MANet、UNetSR的网络定义，以及我们的用于核估计的网络（`./codes/networks/unet_based.py`）。
2. `./codes/optimizers`、`./codes/schedulers`和`./codes/loss_functions`分别包含优化器、学习率调度器和损失函数的定义。
3. `./codes/models`包含模型定义，模型负责整合network、optimizer、scheduler、loss function，定义输入输出、如何优化、学习率如何调度。
4. `./codes/datasets`仅包含数据集的定义。
其中HrLrKernelFromBioSR的配置项比较多。下面选择一些进行注释：
   1. **概述**。在`BioDataset/Mixed`中有一系列SIM GT图片，分别是CCPs、ER、Microtubules、F-actin这四种结构的，各有约50张，
   命名方式统一为`img_xx_y.png`，x代表序号（01, 02, ...），y代表结构（1 for CCPs, 2 for ER, 3 for Microtubules, 4 for F-actin）。
   本数据集的功能是，从`BioDataset/Mixed`中选择一张图片，经过rotate、crop、flip，获得`hr_size`尺寸的High Resolution图片，
   然后根据`psf_settings`生成一个PSF，接着将HR和PSF卷积，再加噪声降采样，获得Low Resolution图片。
   最后，将HR、LR、PSF以及其他相关信息呈递出去。
   2. `is_train`：True则generate data while training networks，每次呈递的HR、LR、PSF均不同，带有data augmentation (rotate、crop、flip)；
   False则固化数据，每次呈递的HR、LR、PSF均相同（LR会由于噪声的影响而有些许不同）。
   3. `preload_data`：若给定路径，则加载指定的.mat文件中的HR、LR、PSF；仅当`is_train`为False时才可以指定此路径。
   4. `gpu_id`：若为None，则使用cpu处理数据；否则使用gpu处理。一般建议使用cpu。
   5. `repeat`：倍增一个epoch的数据量。用来减小训练时在epoch之间切换的时间开销。
   6. `img_filter`
      1. `img_root`：即`BioDataset/Mixed`的具体路径。
      2. `structure_selected`：选择CCPs、ER、Microtubules、F-actin中的哪几种结构。
      3. `included_idx`：即`img_xx_y.png`中的序号xx的取值范围。
   7. `hr_crop`
      1. `mode`：若为random，则随机进行rotate、crop、flip；若为constant，则以`center_pos`为中心进行crop；若为scan，
      则按照`scan_shape: [M, N]`，将原始SIM GT图片分割成M x N块。
      2. `hr_size`：最后获得的HR的尺寸。
   8. `scale`：从HR到LR的降采样倍数，必须为正整数。
   9. `img_signal`：信号强度。背景噪声强度设定为100。
   10. `psf_settings`
       1. `type`：若为Gaussian，则生成高斯核；若为Zernike，则生成Zernike PSF。
       2. `kernel_size`：以pixel为单位的PSF边长。
   11. `sup_phaseZ`：决定呈递的PSF是由哪些Zernike多项式阶数生成出来的。此项基本没啥用，保持all即可。
   12. `padding`：HR和PSF卷积时的相关配置
   13. `loader_settings`：torch.utils.data.DataLoader的相关配置。
5. 上面的提及的文件夹中，均有`__init__.py`文件，包含`get_dataloader`、`get_loss_function`、`get_model`等函数，
用于接收一个option字典，返回相应的object。
6. `./codes/trains`包含所有模型的训练代码。比如说，想训练SFTMD，只需要在命令行运行
`python ./codes/trains/train_SFTMD.py --opt ./options/trains/train_SFTMD.yaml`，所有训练配置项只需在.yaml文件中修改。
7. `./codes/tests`包含所有模型的测试代码。比如说，想测试SFTMD，只需要在命令行运行
`python ./codes/tests/test_SFTMD.py --opt ./options/tests/test_SFTMD.yaml`，所有测试配置项只需在.yaml文件中修改。
8. `./codes/utils`中都是一些工具函数。

本项目使用Wandb记录训练过程的数据。

## Demonstration

### Network Architecture

![](./figures/Network%20Architecture.png "Network Architecture")

### Kernel Comparison

![](./figures/Kernel%20Comparison.png "Kernel Comparison")

### SR Comparison

![](./figures/SR%20Comparison.png "SR Comparison")

### RL Deconv Comparison

![](./figures/RL%20Deconv%20Comparison.png "RL Deconv Comparison")
