import torch
from matplotlib import pyplot as plt
from _tools import cv_frame as cv
from _tools import mini_tool as tool
from d2l import torch as d2l

img = plt.imread('../_data/img/catdog.jpg')
torch.set_printoptions(2)


def bounding_boxes():
    fig = plt.imshow(img)
    dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
    # 添加矩形框
    fig.axes.add_patch(cv.bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(cv.bbox_to_rect(cat_bbox, 'red'))
    plt.show()


def anchor_boxes():
    h, w = img.shape[0:2]
    boxes = cv.generate_multibox(torch.zeros(h, w), sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = boxes.reshape(h, w, 5, 4)
    fig = plt.imshow(img)
    bbox_scale = torch.tensor((w, h, w, h))
    cv.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                   ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
    plt.show()


def labeling_train():
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88, 0.98],
                            [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]])
    labels = cv.label_multibox(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
    print(labels)


def predict():
    h, w = img.shape[0:2]
    bbox_scale = torch.tensor((w, h, w, h))
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4,  # Predicted probability for background
                              [0.9, 0.8, 0.7, 0.1],  # Predicted probability for dog
                              [0.1, 0.2, 0.3, 0.9]])  # Predicted probability for cat
    output = cv.multibox_detection(cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0), anchors.unsqueeze(dim=0),
                                   nms_threshold=0.5)
    fig = plt.imshow(img)
    for box in output[0].detach().numpy():
        if box[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(box[0])] + str(box[1])
        cv.show_bboxes(fig.axes, [torch.tensor(box[2:]) * bbox_scale], label)
    plt.show()


def display_anchors(fmap_w, fmap_h, s):
    h, w = img.shape[0:2]
    # 用小的特征图来均匀采样锚框中心
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = cv.generate_multibox(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    cv.show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)
    plt.show()


if __name__ == '__main__':
    predict()
