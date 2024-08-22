import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel

import utils
from utils import CONFIG
import networks
import numpy as np
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# def visualize_feature_maps(x):
#     """
#     可视化输入的特征图，x 是形状为 (B, C, H, W) 的特征图张量。
#     """
#     B, C, H, W = x.shape
#     feature_maps = x.squeeze().cpu().detach().numpy()
#     # 归一化特征图
#     min_value = feature_maps.min()
#     max_value = feature_maps.max()
#     normalized_feature_maps = (feature_maps - min_value) / (max_value - min_value)
#     # 将归一化的特征图乘以 255
#     scaled_feature_maps = (normalized_feature_maps * 255).astype(np.uint8)
#     # 创建一个大的图像，用于显示所有通道的特征图
#     num_channels = scaled_feature_maps.shape[0]
#     num_rows = 16  # 假设每行显示 16 个通道
#     num_cols = num_channels // num_rows + (num_channels % num_rows > 0)
#     # 创建一个大图像，初始化为全白
#     large_image = np.ones((num_rows * H, num_cols * W), dtype=np.uint8) * 255
#     # 将每个通道的特征图放在大图像的不同位置
#     for channel in range(num_channels):
#         row = channel // num_cols
#         col = channel % num_cols
#         large_image[row * H:(row + 1) * H, col * W:(col + 1) * W] = scaled_feature_maps[channel]
#
#     # 可视化合并后的特征图
#     plt.figure(figsize=(12, 12))  # 调整图像大小
#     plt.imshow(large_image, cmap='viridis')  # 使用合适的 colormap
#     plt.axis('off')  # 隐藏坐标轴
#     plt.title('Merged Feature Maps')
#     plt.show()
#
# def create_outs_block(x, in_channels, out_channels, kernel_size=3, norm_layer=nn.BatchNorm2d, leaky_relu=nn.LeakyReLU(0.2, inplace=True)):
#     out = nn.Sequential(
#         nn.Conv2d(in_channels, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
#         norm_layer(32),
#         leaky_relu,
#         nn.Conv2d(32, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
#     )
#     out.cuda()
#     return out(x)
# def create_all_outs_blocks(input_tensor, kernel_size=3):
#     input_tensor1,input_tensor2,input_tensor3,input_tensor4,input_tensor5=input_tensor[1:]
#     outs1 = create_outs_block(input_tensor1, 32, 3, kernel_size=kernel_size)
#     outs2 = create_outs_block(input_tensor2, 64, 3, kernel_size=kernel_size)
#     outs3 = create_outs_block(input_tensor3, 128, 3, kernel_size=kernel_size)
#     outs4 = create_outs_block(input_tensor4, 256, 3, kernel_size=kernel_size)
#     outs5 = create_outs_block(input_tensor5, 512, 3, kernel_size=kernel_size)
#
#     return outs1, outs2, outs3, outs4, outs5


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16
import warnings

warnings.filterwarnings('ignore')
#
# # 计算特征提取模块的感知损失
# def vgg16_loss(feature_module,loss_func,y,y_):
#     out=feature_module(y)
#     out_=feature_module(y_)
#     loss=loss_func(out,out_)
#     return loss
#
# # 获取指定的特征提取模块
# def get_feature_module(layer_index,device=None):
#     vgg = vgg16(pretrained=True, progress=True).features
#     vgg.eval()
#
#     # 冻结参数
#     for parm in vgg.parameters():
#         parm.requires_grad = False
#
#     feature_module = vgg[0:layer_index + 1]
#     feature_module.to(device)
#     return feature_module
#
#
# # 计算指定的组合模块的感知损失
# class PerceptualLoss(nn.Module):
#     def __init__(self,layer_indexs = [3, 8, 15, 22]):
#         super(PerceptualLoss, self).__init__()
#         loss_func = nn.MSELoss()
#         self.creation=loss_func
#         self.layer_indexs=layer_indexs
#         self.device=device
#
#     def forward(self,y,y_):
#         loss=0
#         for index in self.layer_indexs:
#             feature_module=get_feature_module(index,self.device)
#             loss+=vgg16_loss(feature_module,self.creation,y,y_)
#         return loss
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# layer_indexs = [3, 3, 3, 3]
# loss_func = nn.MSELoss()
# creation = PerceptualLoss(layer_indexs)
# creation.cuda()


class Trainer(object):

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 logger):

        cudnn.benchmark = True

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger

        self.model_config = CONFIG.model
        self.train_config = CONFIG.train
        self.log_config = CONFIG.log
        self.loss_dict = {'rec': None,
                          'comp': None,
                          'lap': None,

                          }
        self.test_loss_dict = {'rec': None,
                               'mse': None,
                               'sad': None,
                              }

        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                          [4., 16., 24., 16., 4.],
                                          [6., 24., 36., 24., 6.],
                                          [4., 16., 24., 16., 4.],
                                          [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)

        self.build_model()
        self.resume_step = None
        self.best_loss = {'mse':1e+8, 'sad':1e+8}

        utils.print_network(self.G, CONFIG.version)

    def build_model(self):
        self.G = networks.get_generator()
        self.G.cuda()
        checkpoint='/home/ljh/SLL/data/sll/MattingCode/AAAA/matteformer_fg_alpha/experiments/240530_161338/checkpoints/best_model_mse.pth'
        checkpoint = torch.load(checkpoint)
        self.G.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
        print("fastnet_t1+sa")

        if CONFIG.dist:
            self.logger.info("Using pytorch synced BN")
            self.G = SyncBatchNorm.convert_sync_batchnorm(self.G)

        self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                            lr=self.train_config.G_lr,
                                            betas=[self.train_config.beta1, self.train_config.beta2])

        if CONFIG.dist:
            # SyncBatchNorm only supports DistributedDataParallel with single GPU per process
            self.G = DistributedDataParallel(self.G, device_ids=[CONFIG.local_rank], output_device=CONFIG.local_rank)
        else:
            self.G = nn.DataParallel(self.G)

        self.build_lr_scheduler()

    def build_lr_scheduler(self):
        """Build cosine learning rate scheduler."""
        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                          T_max=self.train_config.total_step
                                                                - self.train_config.warmup_step)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()

    def train(self):
        data_iter = iter(self.train_dataloader)

        if self.train_config.resume_checkpoint:
            start = self.resume_step + 1
        else:
            start = 0

        moving_max_grad = 0
        moving_grad_moment = 0.999

        for step in range(start, self.train_config.total_step + 1):
            try:
                image_dict = next(data_iter)
            except:
                data_iter = iter(self.train_dataloader)
                image_dict = next(data_iter)

            image, alpha, trimap,fggt = image_dict['image'], image_dict['alpha'], image_dict['trimap'],image_dict['fggt']
            image = image.cuda()
            alpha = alpha.cuda()
            trimap = trimap.cuda()
            alpha_three=image_dict['alpha_three']
            fggt=fggt.cuda()
            fg_norm, bg_norm = image_dict['fg'].cuda(), image_dict['bg'].cuda()


            #
            # alpha1 = (alpha*fg_norm).cpu().detach().numpy() * 255
            # img = np.transpose(alpha1, (0, 2, 3, 1))
            # img = img[0, :, :, :]
            # # 显示图像
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(img))
            # plt.show()
            #
            # alpha1=fg_norm.cpu().detach().numpy()*255
            # img = np.transpose(alpha1, (0, 2, 3, 1))
            # img = img[0, :, :, :]
            # # 显示图像
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(img))
            # plt.show()
            #
            # alpha1 = image.cpu().detach().numpy() * 255
            # img = np.transpose(alpha1, (0, 2, 3, 1))
            # img = img[0, :, :, :]
            # # 显示图像
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(img))
            # plt.show()

            self.G.train()
            loss = 0

            """===== Update Learning Rate ====="""
            if step < self.train_config.warmup_step and self.train_config.resume_checkpoint is None:
                cur_G_lr = utils.warmup_lr(self.train_config.G_lr, step + 1, self.train_config.warmup_step)
                utils.update_lr(cur_G_lr, self.G_optimizer)

            else:
                self.G_scheduler.step()
                cur_G_lr = self.G_scheduler.get_lr()[0]

            """===== Forward G ====="""
            pred = self.G(image, alpha_three,alpha)
            # outs1, outs2, outs3, outs4, outs5 =outs[1:]
            # outs_lable1, outs_lable2, outs_lable3, outs_lable4, outs_lable5 =outs_lable[1:]
            # visualize_feature_maps(outs1)
            # visualize_feature_maps(outs_lable1)
            # visualize_feature_maps(fggt)
            # outs1, outs2, outs3, outs4, outs5=create_all_outs_blocks(outs,3)
            # outs_lable1, outs_lable2, outs_lable3, outs_lable4, outs_lable5 = create_all_outs_blocks(outs_lable, 3)

            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

            weight_os8 = utils.get_unknown_tensor(trimap)
            weight_os8[...] = 1

            if step < self.train_config.warmup_step:
                weight_os4 = utils.get_unknown_tensor(trimap)
                weight_os1 = utils.get_unknown_tensor(trimap)

            elif step < self.train_config.warmup_step * 3:
                if random.randint(0, 1) == 0:
                    weight_os4 = utils.get_unknown_tensor(trimap)
                    weight_os1 = utils.get_unknown_tensor(trimap)
                else:
                    weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                    alpha_pred_os4[weight_os4 == 0] = alpha_pred_os8[weight_os4 == 0]
                    weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
                    alpha_pred_os1[weight_os1 == 0] = alpha_pred_os4[weight_os1 == 0]
            else:
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                alpha_pred_os4[weight_os4 == 0] = alpha_pred_os8[weight_os4 == 0]
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
                alpha_pred_os1[weight_os1 == 0] = alpha_pred_os4[weight_os1 == 0]

            # alpha1 = (alpha_pred_os1 * alpha).cpu().detach().numpy() * 255
            # img = np.transpose(alpha1, (0, 2, 3, 1))
            # img = img[0, :, :, :]
            # # 显示图像
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(img))
            # plt.show()
            #
            # alpha1 = (image).cpu().detach().numpy() * 255
            # img = np.transpose(alpha1, (0, 2, 3, 1))
            # img = img[0, :, :, :]
            # # 显示图像
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(img))
            # plt.show()
            #
            # alpha1 = (fggt).cpu().detach().numpy() * 255
            # img = np.transpose(alpha1, (0, 2, 3, 1))
            # img = img[0, :, :, :]
            # # 显示图像
            # import matplotlib.pyplot as plt
            # plt.imshow(np.uint8(img))
            # plt.show()

            """===== Calculate Loss ====="""
            # if self.train_config.rec_weight > 0:
            #     self.loss_dict['rec'] = (self.regression_loss(alpha_pred_os1, fggt, loss_type='l1', weight=weight_os1) * 2 + \
            #                              self.regression_loss(alpha_pred_os4, fggt, loss_type='l1', weight=weight_os4) * 1 + \
            #                              self.regression_loss(alpha_pred_os8, fggt, loss_type='l1', weight=weight_os8) * 1) / 5.0 * self.train_config.rec_weight
            # if self.train_config.comp_weight > 0:
            #     self.loss_dict['comp'] = (self.composition_loss(alpha_pred_os1, alpha, bg_norm, fggt, weight=weight_os1) * 2 + \
            #                               self.composition_loss(alpha_pred_os4, alpha, bg_norm, fggt, weight=weight_os4) * 1 + \
            #                               self.composition_loss(alpha_pred_os8, alpha, bg_norm, fggt, weight=weight_os8) * 1) / 5.0 * self.train_config.comp_weight

            # if self.train_config.lap_weight > 0:
            #     self.loss_dict['lap'] = ((self.lap_loss(logit=alpha_pred_os1[:, 0, :, :].unsqueeze(1), target=fggt[:, 0, :, :].unsqueeze(1), gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
            #                              self.lap_loss(logit=alpha_pred_os4[:, 0, :, :].unsqueeze(1), target=fggt[:, 0, :, :].unsqueeze(1), gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os4) * 1 + \
            #                              self.lap_loss(logit=alpha_pred_os8[:, 0, :, :].unsqueeze(1), target=fggt[:, 0, :, :].unsqueeze(1), gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight + \
            #                             (self.lap_loss(logit=alpha_pred_os1[:, 1, :, :].unsqueeze(1), target=fggt[:, 1, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
            #                              self.lap_loss(logit=alpha_pred_os4[:, 1, :, :].unsqueeze(1), target=fggt[:, 1, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1',weight=weight_os4) * 1 + \
            #                              self.lap_loss(logit=alpha_pred_os8[:, 1, :, :].unsqueeze(1), target=fggt[:, 1, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1',weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight +\
            #                             (self.lap_loss(logit=alpha_pred_os1[:, 2, :, :].unsqueeze(1), target=fggt[:, 2, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
            #                              self.lap_loss(logit=alpha_pred_os4[:, 2, :, :].unsqueeze(1), target=fggt[:, 2, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os4) * 1 + \
            #                              self.lap_loss(logit=alpha_pred_os8[:, 2, :, :].unsqueeze(1), target=fggt[:, 2, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1',weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight )/3.0


            if self.train_config.rec_weight > 0:
                self.loss_dict['rec'] = (self.regression_loss(alpha_pred_os1, fggt, loss_type='l1', weight=weight_os1) * 2 + \
                                         self.regression_loss(alpha_pred_os4, fggt, loss_type='l1', weight=weight_os4) * 1 + \
                                         self.regression_loss(alpha_pred_os8, fggt, loss_type='l1', weight=weight_os8) * 1) / 5.0 * self.train_config.rec_weight

            if self.train_config.comp_weight > 0:
                self.loss_dict['comp'] = (self.composition_loss(alpha_pred_os1, alpha, bg_norm, fggt, weight=weight_os1) * 2 + \
                                          self.composition_loss(alpha_pred_os4, alpha, bg_norm, fggt, weight=weight_os4) * 1 + \
                                          self.composition_loss(alpha_pred_os8, alpha, bg_norm, fggt, weight=weight_os8) * 1) / 5.0 * self.train_config.comp_weight

            if self.train_config.lap_weight > 0:
                self.loss_dict['lap'] = ((self.lap_loss(logit=(alpha_pred_os1)[:, 0, :, :].unsqueeze(1), target=(fggt)[:, 0, :, :].unsqueeze(1), gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
                                         self.lap_loss(logit=(alpha_pred_os4)[:, 0, :, :].unsqueeze(1), target=(fggt)[:, 0, :, :].unsqueeze(1), gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os4) * 1 + \
                                         self.lap_loss(logit=(alpha_pred_os8)[:, 0, :, :].unsqueeze(1), target=(fggt)[:, 0, :, :].unsqueeze(1), gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight + \
                                        (self.lap_loss(logit=(alpha_pred_os1)[:, 1, :, :].unsqueeze(1), target=(fggt)[:, 1, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
                                         self.lap_loss(logit=(alpha_pred_os4)[:, 1, :, :].unsqueeze(1), target=(fggt)[:, 1, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1',weight=weight_os4) * 1 + \
                                         self.lap_loss(logit=(alpha_pred_os8)[:, 1, :, :].unsqueeze(1), target=(fggt)[:, 1, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1',weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight +\
                                        (self.lap_loss(logit=(alpha_pred_os1)[:, 2, :, :].unsqueeze(1), target=(fggt)[:, 2, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
                                         self.lap_loss(logit=(alpha_pred_os4)[:, 2, :, :].unsqueeze(1), target=(fggt)[:, 2, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1', weight=weight_os4) * 1 + \
                                         self.lap_loss(logit=(alpha_pred_os8)[:, 2, :, :].unsqueeze(1), target=(fggt)[:, 2, :, :].unsqueeze(1),gauss_filter=self.gauss_filter, loss_type='l1',weight=weight_os8) * 1) / 5.0 * self.train_config.lap_weight )/3.0

            # # self.loss_dict['tezheng']=(creation(outs1,outs_lable1)*0.05+creation(outs2,outs_lable2)*0.1+creation(outs3,outs_lable3)*0.2
            #                            +creation(outs4,outs_lable4)*0.4+creation(outs5,outs_lable5)*0.8)*0.05
            # self.loss_dict['tezheng'] = (self.regression_loss(outs1,outs_lable1)*0.05+self.regression_loss(outs2,outs_lable2)*0.1+self.regression_loss(outs3,outs_lable3)*0.2
            #                             +self.regression_loss(outs4,outs_lable4)*0.4+self.regression_loss(outs5,outs_lable5)*0.8)*0.2
            # self.loss_dict['tezheng'] =(creation(alpha_pred_os1,fggt)+creation(alpha_pred_os4,fggt)+creation(alpha_pred_os8,fggt))*0.1

            for loss_key in self.loss_dict.keys():
                if self.loss_dict[loss_key] is not None and loss_key in ['rec', 'comp', 'lap','tezheng']:
                    loss += self.loss_dict[loss_key]

            """===== Back Propagate ====="""
            self.reset_grad()
            loss.backward()

            """===== Clip Large Gradient ====="""
            if self.train_config.clip_grad:
                if moving_max_grad == 0:
                    moving_max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 1e+6)
                    max_grad = moving_max_grad
                else:
                    max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 2 * moving_max_grad)
                    moving_max_grad = moving_max_grad * moving_grad_moment + max_grad * (1 - moving_grad_moment)

            """===== Update Parameters ====="""
            self.G_optimizer.step()

            """===== Write Log ====="""
            # stdout log
            if step % self.log_config.logging_step == 0:
                self.write_log(loss, step, image, trimap, cur_G_lr)

            """===== TEST ====="""
            if ((step % self.train_config.val_step) == 0 or step == self.train_config.total_step) and step > start:
                self.test(step, start)

    def test(self, step, start):
        self.G.eval()
        test_loss = 0
        log_info = ""

        self.test_loss_dict['mse'] = 0
        self.test_loss_dict['sad'] = 0
        for loss_key in self.loss_dict.keys():
            if loss_key in self.test_loss_dict and self.loss_dict[loss_key] is not None:
                self.test_loss_dict[loss_key] = 0

        with torch.no_grad():
            for idx, image_dict in enumerate(self.test_dataloader):
                image, alpha, trimap ,fggt= image_dict['image'], image_dict['alpha'], image_dict['trimap'],image_dict['fggt']
                alpha_shape = image_dict['alpha_shape']
                image = image.cuda()
                alpha_f=image_dict['alpha_f'].cuda()
                alpha=alpha.cuda()

                alpha_three=image_dict['alpha_three']
                alpha_three = alpha_three.cuda()
                trimap = trimap.cuda()
                fggt=fggt.cuda()

                pred= self.G(image, alpha_three,alpha_f)

                alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
                alpha_pred = alpha_pred_os8.clone().detach()
                weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
                alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
                weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
                alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

                h, w = alpha_shape
                alpha_pred = alpha_pred[..., :h, :w]

                trimap = trimap[..., :h, :w]
                fggt=fggt[..., :h, :w]

                weight = utils.get_unknown_tensor(trimap)  # get unknown region (trimap)
                # weight[...] = 1                          # get whole region

                # value of MSE/SAD here is different from test.py and matlab version
                # self.test_loss_dict['mse'] += self.mse(((alpha_pred * 255.).int()).float() / 255., fggt, weight)
                # # self.test_loss_dict['mse'] += self.mse(alpha_pred, alpha, weight)
                # self.test_loss_dict['sad'] += self.sad(alpha_pred, fggt, weight)
                #
                # if self.train_config.rec_weight > 0:
                #     self.test_loss_dict['rec'] += \
                #         self.regression_loss(alpha_pred, fggt, weight=weight) * self.train_config.rec_weight

                self.test_loss_dict['mse'] += self.mse(((alpha_pred * 255.).int()).float() / 255., fggt, weight)
                # self.test_loss_dict['mse'] += self.mse(alpha_pred, alpha, weight)
                self.test_loss_dict['sad'] += self.sad(alpha_pred, alpha*fggt, weight)

                if self.train_config.rec_weight > 0:
                    self.test_loss_dict['rec'] += \
                        self.regression_loss(alpha_pred, fggt, weight=weight) * self.train_config.rec_weight


        # reduce losses from GPUs
        if CONFIG.dist:
            self.test_loss_dict = utils.reduce_tensor_dict(self.test_loss_dict, mode='mean')

        """===== Write Log ====="""
        # stdout log
        for loss_key in self.test_loss_dict.keys():
            if self.test_loss_dict[loss_key] is not None:
                self.test_loss_dict[loss_key] /= len(self.test_dataloader)
                # logging
                log_info += loss_key.upper() + ": {:.4f} ".format(self.test_loss_dict[loss_key])

                if loss_key in ['rec']:
                    test_loss += self.test_loss_dict[loss_key]

        self.logger.info("TEST: LOSS: {:.4f} ".format(test_loss) + log_info)

        """===== Save Model ====="""
        if (step % self.log_config.checkpoint_step == 0 or step == self.train_config.total_step) \
                and CONFIG.local_rank == 0 and (step > start):
            self.logger.info('Saving the trained models from step {}...'.format(iter))
            self.save_model("latest_model", step, self.best_loss)

            if self.test_loss_dict['mse'] < self.best_loss['mse']:
                self.best_loss['mse'] = self.test_loss_dict['mse']
                self.best_loss['sad'] = self.test_loss_dict['sad']
                self.save_model("best_model_mse", step, self.best_loss)

        torch.cuda.empty_cache()

    def write_log(self, loss, step, image, trimap, cur_G_lr):
        log_info= ''

        # reduce losses from GPUs
        if CONFIG.dist:
            self.loss_dict = utils.reduce_tensor_dict(self.loss_dict, mode='mean')
            loss = utils.reduce_tensor(loss)

        # create logging information
        for loss_key in self.loss_dict.keys():
            if self.loss_dict[loss_key] is not None:
                log_info += loss_key.upper() + ": {:.4f}, ".format(self.loss_dict[loss_key])

        self.logger.debug("Image tensor shape: {}. Trimap tensor shape: {}".format(image.shape, trimap.shape))
        log_info = "[{}/{}], ".format(step, self.train_config.total_step) + log_info
        # log_info += "lr: {:.7f}".format(cur_G_lr)
        log_info += "lr: {:6f}".format(cur_G_lr)
        self.logger.info(log_info)

    def save_model(self, checkpoint_name, iter, loss):
        """Restore the trained generator and discriminator."""
        torch.save({
            'iter': iter,
            'loss': loss,
            'state_dict': self.G.state_dict(),
            'opt_state_dict': self.G_optimizer.state_dict(),
            'lr_state_dict': self.G_scheduler.state_dict()
        }, os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(checkpoint_name)))
        self.logger.info('Saving models in step {} iter... : {}'.format(iter, '{}.pth'.format(checkpoint_name)))

    @staticmethod
    def mse(logit, target, weight):
        # return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        return Trainer.regression_loss(logit, target, loss_type='l2', weight=weight)

    @staticmethod
    def sad(logit, target, weight):
        return F.l1_loss(logit * weight, target * weight, reduction='sum') / 1000

    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None):
        """
        Alpha reconstruction loss
        :param logit:
        :param target:
        :param loss_type: "l1" or "l2"
        :param weight: tensor with shape [N,1,H,W] weights for each pixel
        :return:
        """
        if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
        else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))

    @staticmethod
    def composition_loss(pre_fg, alpha, bg, fggt, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        # merged = pre_fg * alpha + bg * (1 - alpha)
        # merged1 =fggt * alpha + bg * (1 - alpha)
        merged = pre_fg * alpha + bg * (1 - alpha)
        merged1 = fggt * alpha  + bg * (1 - alpha)
        return Trainer.regression_loss(merged, merged1, loss_type=loss_type, weight=weight)

    @staticmethod
    def lap_loss(logit, target, gauss_filter, loss_type='l1', weight=None):
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''

        def conv_gauss(x, kernel):
            x = F.pad(x, (2, 2, 2, 2), mode='reflect')
            x = F.conv2d(x, kernel, groups=x.shape[1])
            return x

        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x, kernel):
            N, C, H, W = x.shape
            cc = torch.cat([x, torch.zeros(N, C, H, W).cuda()], dim=3)
            cc = cc.view(N, C, H * 2, W)
            cc = cc.permute(0, 1, 3, 2)
            cc = torch.cat([cc, torch.zeros(N, C, W, H * 2).cuda()], dim=3)
            cc = cc.view(N, C, W * 2, H * 2)
            x_up = cc.permute(0, 1, 3, 2)
            return conv_gauss(x_up, kernel=4 * gauss_filter)

        def lap_pyramid(x, kernel, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                filtered = conv_gauss(current, kernel)
                down = downsample(filtered)
                up = upsample(down, kernel)
                diff = current - up
                pyr.append(diff)
                current = down
            return pyr

        def weight_pyramid(x, max_levels=3):
            current = x
            pyr = []
            for level in range(max_levels):
                down = downsample(current)
                pyr.append(current)
                current = down
            return pyr

        pyr_logit = lap_pyramid(x=logit, kernel=gauss_filter, max_levels=5)
        pyr_target = lap_pyramid(x=target, kernel=gauss_filter, max_levels=5)
        if weight is not None:
            pyr_weight = weight_pyramid(x=weight, max_levels=5)
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2 ** i) for i, A in
                       enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
        else:
            return sum(Trainer.regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2 ** i) for i, A in
                       enumerate(zip(pyr_logit, pyr_target)))