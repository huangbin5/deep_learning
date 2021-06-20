import torch
from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision import io
from matplotlib import pyplot as plt
import os
import pandas as pd
from _tools.constant import *
from _tools import mini_tool as tool
from _tools import mlp_frame as mlp
from _tools import cnn_frame as cnn


def load_cifar10(is_train, augs, batch_size):
    dataset = datasets.CIFAR10(root="../_data", train=is_train, transform=augs, download=True)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)
    return dataloader


def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(cnn.Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(cnn.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 使用更小的卷积核，移除了max pooling层
    net = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


def train_batch(net, X, y, cross_entropy, optimizer, devices):
    if isinstance(X, list):
        # Required for BERT Fine-tuning
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    loss = cross_entropy(pred, y)
    loss.sum().backward()
    optimizer.step()
    train_loss_sum = loss.sum()
    train_acc_sum = mlp.count_accurate(pred, y)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter, cross_entropy, trainer, num_epochs, devices=tool.try_all_gpus()):
    timer, num_batches = tool.Timer(), len(train_iter)
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                             legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        # (sum of loss, num of accurate, num of examples)
        metric = tool.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            loss, acc = train_batch(net, features, labels, cross_entropy, trainer, devices)
            # todo 后面两个数字一样的？
            metric.add(loss, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = cnn.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
    plt.show()


def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes


def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


# bbox是两个角的坐标
def bbox_to_rect(bbox, color):
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
                         fill=False, edgecolor=color, linewidth=2)


# 得到两个角归一化坐标的bbox
def generate_multibox(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 中心点的坐标，+0.5是因为一个像素的中心是(0.5, 0.5)
    center_h = (torch.arange(in_height, device=device) + 0.5) / in_height
    center_w = (torch.arange(in_width, device=device) + 0.5) / in_width
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # 每个中心点有boxes_per_pixel个锚框
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    # 由包含s_0和r_0的(s, r)对组成。w要乘in_height / in_width是为了得到r的比例
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width  # handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 往左上移动的坐标和往右下移动的坐标
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 将中心点和平移坐标相加得到锚框
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',
                      fontsize=10, color=text_color, bbox=dict(facecolor=color, lw=0))


# 计算boxes1和boxes2中任意两个锚框的相似度
def box_iou(boxes1, boxes2):
    # 计算锚框面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_btm = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 得到交集的宽和高，将小于0的值置为0
    w_h = (right_btm - left_top).clamp(min=0)
    inter = w_h[:, :, 0] * w_h[:, :, 1]
    unioun = area1[:, None] + area2 - inter
    return inter / unioun


def match_anchor_to_bbox(ground_truth, anchors, device):
    num_anchors, num_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # 根据阈值给锚框赋值边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    # 返回非0元素的索引，也就是>=0.5的元素索引
    anchor_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anchor_i] = box_j

    # 不断给最大IoU的锚框赋值
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_boxes,), -1)
    for _ in range(num_boxes):
        max_idx = torch.argmax(jaccard)
        anchor_idx = (max_idx / num_boxes).long()
        box_idx = (max_idx % num_boxes).long()
        anchors_bbox_map[anchor_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anchor_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anchor = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anchor[:, :2]) / c_anchor[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anchor[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset


# ground_truth/label/bounding_box指的是一个东西
def label_multibox(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = match_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        batch_mask.append(bbox_mask.reshape(-1))

        # 背景类别是0，其它类别+1
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        batch_labels.append(class_labels)

        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 未匹配锚框的偏移置为0
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_labels)
    return bbox_offset, bbox_mask, class_labels


def offset_inverse(anchors, offset_preds):
    c_anchor = box_corner_to_center(anchors)
    c_pred_bb_xy = (offset_preds[:, :2] * c_anchor[:, 2:] / 10) + c_anchor[:, :2]
    c_pred_bb_wh = torch.exp(offset_preds[:, 2:] / 5) * c_anchor[:, 2:]
    c_pred_bb = torch.cat((c_pred_bb_xy, c_pred_bb_wh), dim=1)
    return box_center_to_corner(c_pred_bb)


def nms(boxes, prob, iou_threshold):
    B = torch.argsort(prob, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.00999999978):
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        prob, class_id = torch.max(cls_prob[1:], 0)
        bbox_pred = offset_inverse(anchors, offset_pred)
        # 根据边界框和概率执行非极大值抑制
        keep = nms(bbox_pred, prob, nms_threshold)

        # 将所有舍弃的边界框类别设置为背景值
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        class_id[non_keep] = -1
        all_id_sorted = torch.cat((keep, non_keep))
        class_id, prob, bbox_pred = class_id[all_id_sorted], prob[all_id_sorted], bbox_pred[all_id_sorted]

        # 预测概率的阈值进行筛选
        below_min_idx = (prob < pos_threshold)
        class_id[below_min_idx] = -1
        prob[below_min_idx] = 1 - prob[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), prob.unsqueeze(1), bbox_pred), dim=1)
        out.append(pred_info)
    return torch.stack(out)


def read_data_bananas(is_train=True):
    DATA_HUB['banana-detection'] = (DATA_URL + 'banana-detection.zip', '5de26c8fce5ccdea9f91267273464dc968d20d72')
    data_dir = tool.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')))
        # target包含5个数，第1个是label，后面4个是边界框
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    train_iter = data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = data.DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter
