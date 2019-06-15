#coding:utf8

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
import numpy as np


# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

featur_cfg = ''


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = self._make_layers(cfg)
        self._rpn_model()

        size = (7, 7)
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
        self.roi_classifier()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        # layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)]
        return nn.Sequential(*layers)
        # return layers

    def _rpn_model(self, mid_channels=512, in_channels=512, n_anchor=9):
        self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # conv sliding layer
        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()

        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        # classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, data):
        out_map = self.features(data)
        # for layer in self.features:
        #     # print layer
        #     data = layer(data)
        #     # print data.data.shape
        #
        # # out = data.view(data.size(0), -1)
        x = self.rpn_conv(out_map)
        pred_anchor_locs = self.reg_layer(x)  # 回归层，计算有效anchor转为目标框的四个系数
        pred_cls_scores = self.cls_layer(x)  # 分类层，判断该anchor是否可以捕获目标

        return out_map, pred_anchor_locs, pred_cls_scores

    def roi_classifier(self, class_num=20):  # 假设为VOC数据集，共20分类
        # 分类层
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                                   nn.ReLU(),
                                                   nn.Linear(4096, 4096),
                                                   nn.ReLU()])
        self.cls_loc = nn.Linear(4096, (class_num+1) * 4)  # (VOC 20 classes + 1 background. Each will have 4 co-ordinates)
        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()


        self.score = nn.Linear(4096, class_num+1)  # (VOC 20 classes + 1 background)

    def rpn_loss(self, rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_label, weight=10.0):
        # 对与classification我们使用Cross Entropy损失
        gt_rpn_label = torch.autograd.Variable(gt_rpn_label.long())
        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        # print(rpn_cls_loss)  # Variable containing: 0.6931

        # 对于 Regression 我们使用smooth L1 损失
        pos = gt_rpn_label.data > 0  # Regression 损失也被应用在有正标签的边界区域中
        mask = pos.unsqueeze(1).expand_as(rpn_loc)
        # print(mask.shape)  # (22500L, 4L)

        # 现在取有正数标签的边界区域
        mask_pred_loc = rpn_loc[mask].view(-1, 4)
        mask_target_loc = gt_rpn_loc[mask].view(-1, 4)
        # print(mask_pred_loc.shape, mask_target_loc.shape)  # ((18L, 4L), (18L, 4L))

        # regression损失应用如下
        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        # print x.shape  # (18, 4)
        # print (x < 1)
        rpn_loc_loss = ((x < 1) * 0.5 * x ** 2) + ((x >= 1) * (x - 0.5))
        # print rpn_loc_loss.shape  # (18, 4)
        rpn_loc_loss = rpn_loc_loss.sum()  # 1.1628926242031001
        # print rpn_loc_loss
        # print rpn_loc_loss.shape
        # rpn_loc_loss = np.squeeze(rpn_loc_loss)
        # print rpn_loc_loss

        N_reg = (gt_rpn_label > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())

        # print "N_reg: {}, {}".format(N_reg, N_reg.shape)
        rpn_loc_loss = rpn_loc_loss / N_reg
        rpn_loc_loss = np.float32(rpn_loc_loss)
        # rpn_loc_loss = torch.autograd.Variable(torch.from_numpy(rpn_loc_loss))

        rpn_cls_loss = np.squeeze(rpn_cls_loss.data.numpy())
        # print "rpn_cls_loss: {}".format(rpn_cls_loss)  # 0.693146109581
        # print 'rpn_loc_loss: {}'.format(rpn_loc_loss)  # 0.0646051466465
        rpn_loss = rpn_cls_loss + (weight * rpn_loc_loss)
        # print("rpn_loss: {}".format(rpn_loss))  # 1.33919757605
        return rpn_loss

    def roi_loss(self, pre_loc, pre_conf, target_loc, target_conf, weight=10.0):
        # 分类损失
        target_conf = torch.autograd.Variable(target_conf.long())
        pred_conf_loss = torch.nn.functional.cross_entropy(pre_conf, target_conf, ignore_index=-1)
        # print(pred_conf_loss)  # Variable containing:  3.0515

        #  对于 Regression 我们使用smooth L1 损失
        # 用计算RPN网络回归损失的方法计算回归损失
        # pre_loc_loss = REGLoss(pre_loc, target_loc)
        pos = target_conf.data > 0  # Regression 损失也被应用在有正标签的边界区域中
        mask = pos.unsqueeze(1).expand_as(pre_loc)  # (128, 4L)

        # 现在取有正数标签的边界区域
        mask_pred_loc = pre_loc[mask].view(-1, 4)
        mask_target_loc = target_loc[mask].view(-1, 4)
        # print(mask_pred_loc.shape, mask_target_loc.shape)  # ((19L, 4L), (19L, 4L))

        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        # print x.shape  # (19, 4)

        pre_loc_loss = ((x < 1) * 0.5 * x ** 2) + ((x >= 1) * (x - 0.5))
        # print(pre_loc_loss.sum())  # 1.4645805211187053

        N_reg = (target_conf > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())
        pre_loc_loss = pre_loc_loss.sum() / N_reg
        pre_loc_loss = np.float32(pre_loc_loss)
        # print pre_loc_loss  # 0.077294916
        # pre_loc_loss = torch.autograd.Variable(torch.from_numpy(pre_loc_loss))
        # 损失总和
        pred_conf_loss = np.squeeze(pred_conf_loss.data.numpy())
        total_loss = pred_conf_loss + (weight * pre_loc_loss)

        return total_loss


if __name__ == '__main__':
    vgg = VGG()
    print vgg
    data = torch.randn((1, 3, 800, 800))
    print data.shape
    data = torch.autograd.Variable(data)
    out = vgg.forward(data)
    print out.data.shape


