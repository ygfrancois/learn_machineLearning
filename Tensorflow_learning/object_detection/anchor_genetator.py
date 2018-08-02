import numpy as np


def generate_anchors(
        stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchor[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors


def _whctrs(anchor):
    # ctr =  center
    # anchor 包含4个顶点的坐标
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchor(ws, hs, x_ctr, y_xtr):
    # np.newaxis在ndarry里的功能相当于None，给多维数组添加一个轴
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((
        x_ctr - 0.5 * (ws - 1),
        y_xtr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1),
        y_xtr + 0.5 * (hs - 1)
    ))
    return anchors


def _ratio_enum(anchor, ratios):
    # aspect ratio: 纵横比
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws*ratios)
    anchors = _mkanchor(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchor(ws, hs, x_ctr, y_ctr)
    return anchors