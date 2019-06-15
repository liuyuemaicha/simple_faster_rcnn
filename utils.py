# coding:utf8
import sys
import numpy as np
import torch

def init_anchor(img_size=800, sub_sample=16):
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]  # 该尺寸是针对特征图的

    # 一个特征点对应原图片中的16*16个像素点区域, 'img_size // sub_sample'得到特征图的尺寸
    feature_size = (img_size // sub_sample)
    # 这里相当于把图像分割成feature_size*feature_size的网格， 每个网格对应一个特征点。
    # ctr_x， ctr_y: 每个网格的右下方坐标
    ctr_x = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)  # 共feature_size个
    ctr_y = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)  # 共feature_size个
    # print len(ctr_x)  # 50

    index = 0
    # ctr: 每个网格的中心点，一共feature_size*feature_size个网格
    ctr = dict()
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index] = [-1, -1]
            ctr[index][1] = ctr_x[x] - 8  # 右下角坐标 - 8 = 中心坐标
            ctr[index][0] = ctr_y[y] - 8
            index += 1
    # print len(ctr)  # 将原图片分割成50*50=2500(feature_size*feature_size)个区域的中心点

    # 初始化：每个区域有9个anchors候选框，每个候选框的坐标(y1, x1, y2, x2)
    anchors = np.zeros(((feature_size * feature_size * 9), 4))  # (22500, 4)
    index = 0
    # 将候选框的坐标赋值到anchors
    for c in ctr:
        ctr_y, ctr_x = ctr[c]
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                # anchor_scales 是针对特征图的，所以需要乘以下采样"sub_sample"
                h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
                anchors[index, 0] = ctr_y - h / 2.
                anchors[index, 1] = ctr_x - w / 2.
                anchors[index, 2] = ctr_y + h / 2.
                anchors[index, 3] = ctr_x + w / 2.
                index += 1

    # 去除坐标出界的边框，保留图片内的框——图片内框
    valid_anchor_index = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= 800) &
        (anchors[:, 3] <= 800)
    )[0]  # 该函数返回数组中满足条件的index
    # print valid_anchor_index.shape  # (8940,)，表明有8940个框满足条件

    # 获取有效anchor（即边框都在图片内的anchor）的坐标
    valid_anchor_boxes = anchors[valid_anchor_index]
    # print(valid_anchor_boxes.shape)  # (8940, 4)

    return anchors, valid_anchor_boxes, valid_anchor_index


# 计算有效anchor框"valid_anchor_boxes"与目标框"bbox"的IOU
def compute_iou(valid_anchor_boxes, bbox):
    valid_anchor_num = len(valid_anchor_boxes)
    ious = np.empty((valid_anchor_num, 2), dtype=np.float32)
    ious.fill(0)
    for num1, i in enumerate(valid_anchor_boxes):
        ya1, xa1, ya2, xa2 = i
        anchor_area = (ya2 - ya1) * (xa2 - xa1)  # anchor框面积
        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2 - yb1) * (xb2 - xb1)  # 目标框面积
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)  # anchor框和目标框的相交面积
                iou = iter_area / (anchor_area + box_area - iter_area)  # IOU计算
            else:
                iou = 0.

            ious[num1, num2] = iou

    return ious


def get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7,neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256):
    gt_argmax_ious = ious.argmax(axis=0)  # 找出每个目标实体框最大IOU的anchor框index，共2个, 与图片内目标框数量一致
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # 获取每个目标实体框最大IOU的值，与gt_argmax_ious对应, 共2个，与图片内目标框数量一致
    argmax_ious = ious.argmax(axis=1)  # 找出每个anchor框最大IOU的目标框index，共8940个, 每个anchor框都会对应一个最大IOU的目标框
    max_ious = ious[np.arange(valid_anchor_len), argmax_ious]  # 获取每个anchor框的最大IOU值， 与argmax_ious对应, 每个anchor框内都会有一个最大值

    gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # 根据上面获取的目标最大IOU值，获取等于该值的index
    # print gt_argmax_ious.shape  # (18,) 共计18个

    label = np.empty((valid_anchor_len,), dtype=np.int32)
    label.fill(-1)
    # print label.shape  # (8940,)
    label[max_ious < neg_iou_threshold] = 0  # anchor框内最大IOU值小于neg_iou_threshold，设为0
    label[gt_argmax_ious] = 1  # anchor框有全局最大IOU值，设为1
    label[max_ious >= pos_iou_threshold] = 1  # anchor框内最大IOU值大于等于pos_iou_threshold，设为1

    n_pos = pos_ratio * n_sample  # 正例样本数

    # 随机获取n_pos个正例，
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1

    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]

    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1

    return label, argmax_ious


def get_predict_bbox(anchors, pred_anchor_locs, objectness_score, n_train_pre_nms=12000, min_size=16):
    # 转换anchor格式从 y1, x1, y2, x2 到 ctr_x, ctr_y, h, w ：
    anc_height = anchors[:, 2] - anchors[:, 0]
    anc_width = anchors[:, 3] - anchors[:, 1]
    anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

    # 根据预测的四个系数，将anchor框通过平移和缩放转化为预测的目标框
    pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
    objectness_score_numpy = objectness_score[0].data.numpy()
    dy = pred_anchor_locs_numpy[:, 0::4]
    dx = pred_anchor_locs_numpy[:, 1::4]
    dh = pred_anchor_locs_numpy[:, 2::4]
    dw = pred_anchor_locs_numpy[:, 3::4]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    #  将预测的目标框转换为[y1, x1, y2, x2]格式
    roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
    roi[:, 0::4] = ctr_y - 0.5 * h
    roi[:, 1::4] = ctr_x - 0.5 * w
    roi[:, 2::4] = ctr_y + 0.5 * h
    roi[:, 3::4] = ctr_x + 0.5 * w

    # 保证预测框的坐标全部落在图片中，y1,y2在（0, img_size[0]）之间, x1,x2在（0, img_size[1]）之间
    img_size = (800, 800)  # Image size
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])
    # print(roi.shape)  # (22500, 4)

    #  去除高度或宽度 < threshold的预测框 （疑问：这样会不会忽略小目标）
    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :]
    score = objectness_score_numpy[keep]

    # 按分数从高到低排序所有的（proposal, score）对
    order = score.ravel().argsort()[::-1]   # (22500,)
    # 取前几个预测框pre_nms_topN(如训练时12000，测试时300)
    order = order[:n_train_pre_nms]

    return roi, score, order

# torch.masked_select()

def nms(roi, score, order, nms_thresh=0.7, n_train_post_nms=2000):
    # nms（非极大抑制）计算： (去除和极大值anchor框IOU大于0.7的框——即去除相交的框，保留score大，且基本无相交的框)
    roi = roi[order, :]  # (12000, 4)
    score = score[order]
    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = score.argsort()[::-1]
    # print score
    # print order
    keep = []
    while order.size > 0:
        # print order
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # print ovr
        inds = np.where(ovr <= nms_thresh)[0]
        # print inds
        order = order[inds + 1]  # 这里加1是因为在计算IOU时，把序列的第一个忽略了（如上面的order[1:]）

    keep = keep[:n_train_post_nms]  # while training/testing , use accordingly
    roi = roi[keep]  # the final region proposals（region proposals表示预测目标框）
    # print roi.shape  # (1758, 4)
    return roi


def get_propose_target(roi, bbox, labels, n_sample=128, pos_ratio=0.25,
                       pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo = 0.0):
    # Proposal targets
    # 找到每个ground-truth目标（真实目标框bbox）与region proposal（预测目标框roi）的iou
    ious = compute_iou(roi, bbox)
    # print(ious.shape)  # (1758, 2)

    # 找到与每个region proposal具有较高IoU的ground truth，并且找到最大的IoU
    gt_assignment = ious.argmax(axis=1)
    max_iou = ious.max(axis=1)
    # print(gt_assignment)  # [0 0 1 ... 0 0 0]
    # print(max_iou)  # [0.17802152 0.17926688 0.04676317 ... 0.         0.         0.        ]

    # 为每个proposal分配标签：
    gt_roi_label = labels[gt_assignment]
    # print(gt_roi_label)  # [6 6 8 ... 6 6 6]

    # 希望只保留n_sample*pos_ratio（128*0.25=32）个前景样本，因此如果只得到少于32个正样本，保持原状。
    # 如果得到多余32个前景目标，从中采样32个样本
    pos_roi_per_image = n_sample*pos_ratio
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
    # print(pos_roi_per_this_image)
    # print(pos_index)  # 19

    # 针对负[背景]region proposal进行相似处理
    neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
    # print(neg_roi_per_this_image)
    # print(neg_index)  # 109

    keep_index = np.append(pos_index, neg_index)
    gt_roi_labels = gt_roi_label[keep_index]
    gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
    sample_roi = roi[keep_index]  # 预测框
    # print(sample_roi.shape)  # (128, 4)
    return sample_roi, keep_index, gt_assignment, gt_roi_labels


def get_coefficient(anchor, bbox):
    # 根据上面得到的预测框和与之对应的目标框，计算4维参数（平移参数：dy, dx； 缩放参数：dh, dw）
    height = anchor[:, 2] - anchor[:, 0]
    width = anchor[:, 3] - anchor[:, 1]
    ctr_y = anchor[:, 0] + 0.5 * height
    ctr_x = anchor[:, 1] + 0.5 * width
    base_height = bbox[:, 2] - bbox[:, 0]
    base_width = bbox[:, 3] - bbox[:, 1]
    base_ctr_y = bbox[:, 0] + 0.5 * base_height
    base_ctr_x = bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
    # print(gt_roi_locs.shape)

    return gt_roi_locs