import torch
from torch import nn
from torch.nn import functional as F
from torchvision import io
from matplotlib import pyplot as plt
from _tools import cv_frame as cv
from _tools import mini_tool as tool


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = cv.generate_multibox(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


def concat_preds(preds):
    def flatten_pred(pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    return torch.cat([flatten_pred(p) for p in preds], dim=1)


class TinySSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 包含s0或r0的(s, r)组合
        num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        for i in range(5):
            # 1个基本网络 + 3个下采样模块 + 1个全局最大池化
            setattr(self, f'blk_{i}', get_blk(i))
            # 5个模块都会衍生出1个计算类别的网络 + 1个计算偏移的网络
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def train(net):
    num_epochs, timer = 20, tool.Timer()
    animator = tool.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    net = net.to(device)
    for epoch in range(num_epochs):
        print('epoch {}/{} begin...'.format(epoch + 1, num_epochs))
        # (cls_loss, num_of_cls_pred, bbox_loss, num_of_bbox_pred)
        metric = tool.Accumulator(4)
        net.train()
        for features, labels in train_iter:
            timer.start()
            optimizer.zero_grad()
            X, Y = features.to(device), labels.to(device)
            # 产生锚框并预测类别和偏移
            anchors, cls_preds, bbox_preds = net(X)
            # 标记类别和偏移
            bbox_labels, bbox_masks, cls_labels = cv.label_multibox(anchors, Y)
            loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            loss.mean().backward()
            optimizer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')
    plt.show()


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = cv.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display(img, output, threshold):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        cv.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.show()


if __name__ == '__main__':
    batch_size, device = 32, tool.try_gpu()
    train_iter, _ = cv.load_data_bananas(batch_size)
    idx_to_in_channels = [64, 128, 128, 128, 128]
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    net = TinySSD(num_classes=1)
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    # train(net)
    X = io.read_image('../_data/img/banana.jpg').unsqueeze(0).float()
    output = predict(X)
    img = X.squeeze(0).permute(1, 2, 0).long()
    display(img, output.cpu(), threshold=0.9)
