import skimage.io
import skimage.feature
import skimage.color
import skimage.util
import skimage.segmentation
import numpy as np
from matplotlib import pyplot as plt


def _generate_segments(img, scale, sigma, min_size):
    """ 区域分割 """
    # segment_mask.shape = (height, width)：分割后所在的的区域编号
    segment_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(img),
                                                     scale=scale, sigma=sigma, min_size=min_size)
    # 将segment_mask加到最后一个通道
    img = np.append(img, segment_mask[:, :, None], axis=2).astype(np.float32)
    return img


def _calc_colour_hist(img):
    """
    使用L1-norm归一化获取图像每个颜色通道的25 bins的直方图
    hist.shape = (75,)
    """
    BINS = 25
    hist = np.array([])
    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]
        # np.histogram返回(频数，区间)
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
    hist = hist / len(img)
    return hist


def _calc_texture_gradient(img):
    """
    计算纹理梯度。对每个颜色通道的8个不同方向计算方差σ=1的高斯微分，这里使用LBP替代
    ret.shape = (height, width, 4)
    """
    ret = np.zeros(img.shape)
    for channel in (0, 1, 2):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(img[:, :, channel], 8, 1.0)
    return ret


def _calc_texture_hist(img):
    """
    使用L1-norm归一化获取图像每个颜色通道的每个方向的10 bins的直方图，这样就可以获取到一个240（10x8x3）维的向量？？？
    hist.shape = (10,)
    """
    BINS = 10
    hist = np.array([])
    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (0.0, 1.0))[0]])
    hist = hist / len(img)
    return hist


def _extract_regions(img):
    """
    R: dict 每个区域的信息
        {
            label: {
                min_x:边界框的左上角x坐标,
                min_y:边界框的左上角y坐标,
                max_x:边界框的右下角x坐标,
                max_y:边界框的右下角y坐标,
                labels:区域编号
                size:像素个数,
                hist_c:颜色的直方图,
                hist_t:纹理特征的直方图
            },
            ...
        }
    """
    R = {}

    # pass 1: 遍历每一个像素
    for y, xc in enumerate(img):
        for x, (r, g, b, l) in enumerate(xc):
            if l not in R:
                # 新的区域
                R[l] = {"min_x": 0xffff, "min_y": 0xffff, "max_x": 0, "max_y": 0, "labels": [l]}
            # 更新当前像素所在区域的边界框
            R[l]['min_x'] = min(R[l]['min_x'], x)
            R[l]['min_y'] = min(R[l]['min_y'], y)
            R[l]['max_x'] = max(R[l]['max_x'], x)
            R[l]['max_y'] = max(R[l]['max_y'], y)

    # pass 2: 提取纹理特征。tex_grad.shape = (height, width, 4)
    tex_grad = _calc_texture_gradient(img)

    # pass 3: 计算每一个区域的直方图
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    for k in R.keys():
        # shape = (区域像素数，3)
        masked_pixels = hsv[img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels)  # 候选区域k像素数
        # 获取图像每个颜色通道的25 bins的直方图，得到一个75维的向量
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # 获取图像每个颜色通道的每个方向的10 bins的直方图，得到一个240（10x8x3）维的向量？？？
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R


def _extract_neighbours(regions):
    """
    提取邻居区域对
    """

    # 判断两个区域是否相交
    def intersect(a, b):
        return (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]) \
               or (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) \
               or (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) \
               or (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"])

    R = list(regions.items())
    neighbours = []
    # 每次抽取两个候选区域 两两组合，判断是否相交
    for i, a in enumerate(R[:-1]):
        for b in R[i + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours


def _calc_sim(r1, r2, img_size):
    """
    计算两个区域的相似度，权重系数默认都是1
    """

    def _sim_colour(r1, r2):
        """
        计算颜色相似度: 将每个位置的最小值加起来

        """
        return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])

    def _sim_texture(r1, r2):
        """
        计算纹理特征相似度: 将每个位置的最小值加起来

        """
        return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])

    def _sim_size(r1, r2):
        """
        计算候选区域大小相似度
        """
        return 1.0 - (r1["size"] + r2["size"]) / img_size

    def _sim_fill(r1, r2):
        """
        计算区域的合适度距离
        """
        bbsize = ((max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
                  * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])))
        return 1.0 - (bbsize - r1["size"] - r2["size"]) / img_size

    return _sim_colour(r1, r2) + _sim_texture(r1, r2) + _sim_size(r1, r2) + _sim_fill(r1, r2)


def _merge_regions(r1, r2):
    """
    合并区域对
    """
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "labels": r1["labels"] + r2["labels"],
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size
    }
    return rt


def selective_search(img, scale=1.0, sigma=0.8, min_size=50):
    """
    regions : array of dict
        [
            {
                'rect': (left, top, width, height),
                'labels': [...],
                'size': component_size
            },
            ...
        ]
    """
    assert img.ndim == 3, 'must be 3 dimensions'
    assert img.shape[2] == 3, "must be 3 channels"

    # 图片分割，将候选区域标签合并到最后一个通道上。通道维度[r, g, b, (region)]
    img = _generate_segments(img, scale, sigma, min_size)
    if img is None:
        return None, []
    # 获取每个区域的信息
    R = _extract_regions(img)

    # 提取邻居区域对
    neighbours = _extract_neighbours(R)
    S = {}
    # 计算每一邻居区域对的相似度
    img_size = img.shape[0] * img.shape[1]
    for (al, ar), (bl, br) in neighbours:
        S[(al, bl)] = _calc_sim(ar, br, img_size)

    # 层次搜索，直至相似度集合为空
    while S != {}:
        # 选择相似度最高的区域对(i, j)
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
        # 合并区域对(i, j)
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # 删除与区域对(i, j)相关的相似度
        keys_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del S[k]

        # 计算新区域与其它相邻区域的相似度
        for k in filter(lambda a: a != (i, j), keys_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], img_size)

    # 获取每一个候选区域的的信息  边框、以及候选区域size,标签
    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    # img：基于图的图像分割得到的每个像素的区域
    # regions：Selective Search算法得到的候选区域
    return img, regions


def main():
    img = skimage.io.imread('../_data/img/catdog.jpg')

    '''
    执行selective search，regions格式如下
    [
        {
            'rect': (left, top, width, height),
            'labels': [...],
            'size': component_size
        },
        ...
    ]
    '''
    img, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # 排除重复的候选区
        if r['rect'] in candidates:
            continue
        # 排除小于2000 pixels的候选区域
        if r['size'] < 2000:
            continue
        # 排除长宽比极端的候选区域
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # 在原始图像上绘制候选区域边框
    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img[:, :, :-1].astype(np.uint8))
    for x, y, w, h in candidates:
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
