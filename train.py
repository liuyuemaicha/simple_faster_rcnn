#coding:utf8

import torch
import torchvision
from PIL import Image, ImageDraw
import numpy as np
from model import VGG
import utils


# 大体流程：
# 1. 图像通过vgg获得特征图，
# 2. 特征图通过RPN获得有效anchor的置信度(foreground)和转为预测框的坐标系数
# 3. 特征图和预测框通过ROI Pooling获取固定尺寸的预测目标特征图,即利用预测框，从特征图中把目标抠出来，
#    因为目标尺寸不一，再通过ROI Pooling的方法把目标转为统一的固定尺寸（7*7），这样就可以方便做目标的分类和预测框的修正处理。
# 4. 固定尺寸的预测目标特征图通过类别分类模型（self.score）获取预测框所属的类别，
# 5. 固定尺寸的预测目标特征图通过坐标分类模型(self.cls_loc)获取置信度和预测框修正的坐标系数。

# 假设 图片中的两个目标框"ground-truth"
bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
# 假设 图片中两个目标框分别对应的标签
labels = np.asarray([6, 8], dtype=np.int8)  # 0 represents background

img_tensor = torch.zeros((1, 3, 800, 800)).float()
img_var = torch.autograd.Variable(img_tensor)


# ---------------------step_1: 获取目标anchor的置信度（anchor_conf）和平移缩放系数（anchor_locations）
# 初始化所有anchors, 并找出有效anchors和对应的index
# anchors： (22500, 4)  valid_anchor_boxes： (8940, 4)  valid_anchor_index：8940
anchors, valid_anchor_boxes, valid_anchor_index = utils.init_anchor()
# 计算有效anchors与所有目标框的IOU
# ious：（8940, 2) 每个有效anchor框与目标实体框的IOU
ious = utils.compute_iou(valid_anchor_boxes, bbox)
valid_anchor_len = len(valid_anchor_boxes)
# 在有效框中找到一定比例的正例和负例
label, argmax_ious = utils.get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7,
                                 neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256)
# print np.sum(label == 1)  # 18个正例
# print np.sum(label == 0)  # 256-18=238个负例

# 现在让我们用具有最大iou的ground truth对象为每个anchor box分配位置。
# 注意，我们将为所有有效的anchor box分配anchor locs，而不考虑其标签，稍后在计算损失时，我们可以使用简单的过滤器删除它们。
# 每个有效anchor对应的目标框bbox
max_iou_bbox = bbox[argmax_ious]  # 有效anchor框对应的目标框坐标  (8940, 4)
# print max_iou_bbox.shape  # (8940, 4)，共有8940个有效anchor框，每个anchor有坐标值（y1, x1, y2, x2）
# 为所有有效的anchor_box分配anchor_locs，anchor_locs是每个有效的anchors转为对应目标框（bbox）的平移缩放系数
anchor_locs = utils.get_coefficient(valid_anchor_boxes, max_iou_bbox)
# print(anchor_locs.shape)  # (8940, 4)  4维参数（平移参数：dy, dx； 缩放参数：dh, dw）

# anchor_conf ： 所有anchor框对应的label（-1：无效anchor，0：负例有效anchor，1：正例有效anchor）
anchor_conf = np.empty((len(anchors),), dtype=label.dtype)
anchor_conf.fill(-1)
anchor_conf[valid_anchor_index] = label
print anchor_conf.shape  # 所有anchor对应的label（feature_size*feature_size*9）=》 (22500,)

# anchor_locations： 所有anchor框转为目标实体框的系数，无效anchor系数全部为0，有效anchor有有效系数
anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[valid_anchor_index, :] = anchor_locs
print anchor_locations.shape  # 所有anchor对应的平移缩放系数（feature_size*feature_size*9，4）=》(22500, 4)

# 这里通过候选anchor与目标实体框计算得到anchor框的置信度（anchor_conf）和平移缩放系数（anchor_locations）
# ----------------------


# --------------------step_2: VGG 和 RPN 模型: RPN 预测的是anchor转为目标框的平移缩放系数
vgg = VGG()
# out_map 特征图， # pred_anchor_locs 预测anchor框到目标框转化的系数， pred_anchor_conf 预测anchor框的分数
out_map, pred_anchor_locs, pred_anchor_conf = vgg.forward(img_var)
print out_map.data.shape  # (batch_size, num, feature_size, feature_size) => (1, 512, 50, 50)

# 1. pred_anchor_locs 预测每个anchor框到目标框转化的系数（平移缩放），与 anchor_locations对应
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print(pred_anchor_locs.shape)  # Out: torch.Size([1, 22500, 4])

# 2. 预测anchor框的置信度，每个anchor框都会对应一个置信度，与 anchor_conf对应
pred_anchor_conf = pred_anchor_conf.permute(0, 2, 3, 1).contiguous()
print(pred_anchor_conf.shape)  # Out torch.Size([1, 50, 50, 18])
objectness_score = pred_anchor_conf.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print(objectness_score.shape)  # Out torch.Size([1, 22500])

pred_anchor_conf = pred_anchor_conf.view(1, -1, 2)
print(pred_anchor_conf.shape)  # Out torch.size([1, 22500, 2])
# ---------------------


# ---------------------step_3: RPN 损失 （有效anchor与预测anchor之间的损失--坐标系数损失与置信度损失）
# 从上面step_1中，我们得到了目标anchor信息：
# 目标anchor坐标系数：anchor_locations  (22500, 4)
# 目标anchor置信度：anchor_conf  (22500,)

# 从上面step_2中，我们得到了预测anchor信息：
# RPN网络预测anchor的坐标系数：pred_anchor_locs  (1, 22500, 4)
# RPN网络预测anchor的置信度: pred_anchor_conf  (1, 22500, 2)

# 我们将会从新排列，将输入和输出排成一行
rpn_anchor_loc = pred_anchor_locs[0]
rpn_anchor_conf = pred_anchor_conf[0]
anchor_locations = torch.from_numpy(anchor_locations)
anchor_conf = torch.from_numpy(anchor_conf)
print(rpn_anchor_loc.shape, rpn_anchor_conf.shape, anchor_locations.shape, anchor_conf.shape)
# torch.Size([22500, 4]) torch.Size([22500, 2]) torch.Size([22500, 4]) torch.Size([22500])

rpn_loss = vgg.roi_loss(rpn_anchor_loc, rpn_anchor_conf, anchor_locations, anchor_conf, weight=10.0)
print("rpn_loss: {}".format(rpn_loss))  # 1.33919
# ---------------------


# ---------------------step_4: 根据anchor和预测anchor系数，计算预测框（roi）和预测框的坐标系数(roi_locs)，
# ---------------------并得到每个预测框的所属类别label(roi_labels)
# 通过anchors框和模型预测的平移缩放系数，得到预测框ROI；再通过预测的分值和阈值进行过滤精简
roi, score, order = utils.get_predict_bbox(anchors, pred_anchor_locs, objectness_score,
                                           n_train_pre_nms=12000, min_size=16)

# 得到的预测框（ROI）还会有大量重叠，再通过NMS（非极大抑制）做进一步的过滤精简
roi = utils.nms(roi, score, order, nms_thresh=0.7, n_train_post_nms=2000)


# 根据预测框ROI与目标框BBox的IOU，得到每个预测框所要预测的目标框（预测框与哪个目标框的IOU大，就代表预测哪个目标）；
# 并根据IOU对ROI做进一步过滤，并划分正负样例。
sample_roi, keep_index, gt_assignment, roi_labels = utils.get_propose_target(roi, bbox, labels,
                                                                                n_sample=128,
                                                                                pos_ratio=0.25,
                                                                                pos_iou_thresh=0.5,
                                                                                neg_iou_thresh_hi=0.5,
                                                                                neg_iou_thresh_lo=0.0)
# print(sample_roi.shape)  # (128, 4)
# 预测框对应的目标框 bbox_for_sampled_roi
bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]  # 目标框
print(bbox_for_sampled_roi.shape)  # (128, 4)
# 预测框（ROI）转目标框的真实系数
roi_locs = utils.get_coefficient(sample_roi, bbox_for_sampled_roi)
# ---------------------


# ---------------------step_5: ROI Pooling：
# 这一步做了两件事：
# 一是从特征图中根据ROI把相应的预测目标框抠出来(im)
# 二是将抠出来的预测目标框通过adaptive_max_pool方法，输出为固定尺寸(512, 7, 7)，方便后续的批处理
# 这样的特点：
# 一是并没有在输入图像上预测，而是在VGG模型的输出特征图上进行预测，这样减少了计算量；
# 二是因为目标实体尺寸多种多样，通过ROI Pooling方法将输出统一为固定尺寸(512, 7, 7)，方便进行批处理，
# sample_roi：预测的有效框 (128, 4)
rois = torch.from_numpy(sample_roi).float()
# roi_indices：添加图像的索引[这里我们只有一个图像，其索引号为0]
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
print(rois.shape, roi_indices.shape)  # torch.Size([128, 4]) torch.Size([128])

# 将图像的索引号和预测的有效框进行合并, 这样我们将会得到维度是[N, 5]  5=>(index, x1, y1, x2, y2)的张量
indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)  # torch.Size([128, 5])

output = []
rois = indices_and_rois.float()
rois[:, 1:].mul_(1/16.0)  # 对预测框进行下采样，匹配特征图out_map
rois = rois.long()
num_rois = rois.size(0)
# out_map: (batch_size, num, feature_size, feature_size) => (1, 512, 50, 50)
for i in range(num_rois):
   roi = rois[i]
   im_idx = roi[0]  # 图片的索引号
   # 取出索引号是im_idx的图片特征图=》(1, 512, 50, 50)，因为本实例就一张图片，所以操作完后shape并不变
   out_map = out_map.narrow(0, im_idx, 1)
   # 这一步是根据预测框的的x1,y1, x2,y2坐标，从特征图out_map中把目标实体抠出来
   im = out_map[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
   # print im.shape
   # 将抠出来的目标实体im，做adaptive_max_pool计算，最后得到一个固定的尺寸(7,7)== > (512, 7, 7)，方便后面进行批处理
   output.append(vgg.adaptive_max_pool(im)[0].data)
# ---------------------ROI Pooling


# ---------------------step_6: Classification 线性分类，预测预测框的类别，置信度和转为目标框的平移缩放系数（要与RPN区分）
# note: if your pytorch version is 0.3.1, you must run this:
# output = torch.stack(output)
output = torch.cat(output, 0)  # torch.Size([128, 512, 7, 7])
k = output.view(output.size(0), -1)  # [128, 25088]

k = torch.autograd.Variable(k)
k = vgg.roi_head_classifier(k)  # (128, 4096)
# torch.Size([128, 84])  84 ==> (20+1)*4,表示每个框有20个候选类别和一个置信度（假设为VOC数据集，共20分类），4表示坐标信息
pred_roi_locs = vgg.cls_loc(k)
# pred_roi_labels： [128, 21] 表示每个框的类别和置信度
pred_roi_labels = vgg.score(k)
print(pred_roi_locs.data.shape, pred_roi_labels.data.shape)  # torch.Size([128, 84]), torch.Size([128, 21])
# ---------------------Classification


# ---------------------step_7: 分类损失  (有效预测框真实系数与有效预测框的预测系数间损失，其中系数是转为目标框的坐标系数)
# 从上面step_4中，我们得到了预测框转为目标框的目标信息：
# 预测框的坐标系数(roi_locs)：  (128, 4)
# 预测框的所属类别(roi_labels)：(128, )

# 从上面step_6中，我们得到了预测框转为目标框的预测信息：
# 预测框的坐标系数：pred_roi_locs  (128, 84)
# 预测框的所属类别和置信度: pred_roi_labels  (128, 21)


gt_roi_loc = torch.from_numpy(roi_locs)
gt_roi_label = torch.from_numpy(np.float32(roi_labels)).long()
print(gt_roi_loc.shape, gt_roi_label.shape)  # torch.Size([128, 4]) torch.Size([128])

n_sample = pred_roi_locs.shape[0]
roi_loc = pred_roi_locs.view(n_sample, -1, 4)  # (128L, 21L, 4L)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]  # 根据预测框的真实类别，找到真实类别所对应的坐标系数
# print(roi_loc.shape)  # torch.Size([128, 4])

roi_loss = vgg.roi_loss(roi_loc, pred_roi_labels, gt_roi_loc, gt_roi_label, weight=10.0)
print(roi_loss)  # 3.810348778963089


# 整体损失函数
total_loss = rpn_loss + roi_loss
print total_loss  # 5.149546355009079


