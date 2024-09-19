import math
import random

import cv2
import numpy as np

from ..augmentations import box_candidates
from ..general import resample_segments, segment2box


def mixup(im, labels, segments, im2, labels2, segments2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    segments = np.concatenate((segments, segments2), 0)
    return im, labels, segments


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    new_segments = []
    new_bboxes = []
    new_targets = []
    if n:
        segments = resample_segments(segments)  # upsample
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2])  # perspective rescale or affine

            # clip
            x, y = xy[:, 0], xy[:, 1]  # segment xy
            inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
            xy = xy[inside]
            if len(xy) == 0:
                continue
            if len(xy) != len(segment):
                xy = resample_segments([xy])[0]

            new_segments.append(xy)
            
            # sample bbox separately and clip
            xy = np.ones((2, 3))
            xy[:, :2] = targets[i][1:].reshape((2, 2))
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2])
            
            # clip
            x1, y1, x2, y2 = xy.reshape((-1))
            x1, x2 = [min(max(v, 0), width - 1) for v in [x1, x2]]
            y1, y2 = [min(max(v, 0), height - 1) for v in [y1, y2]]
            xy = np.array([x1, y1, x2, y2]).reshape((2, 2))
            
            new_bboxes.append(segment2box(xy, width, height))

            new_targets.append(targets[i])

        # filter candidates
        if len(new_segments) == 0:
            return im, np.empty((0, 5), np.float32), new_segments
        
        new_bboxes = np.array(new_bboxes)
        new_targets = np.array(new_targets)
        
        i = box_candidates(box1=new_targets[:, 1:5].T * s, box2=new_bboxes.T, area_thr=0.01)
        new_targets = new_targets[i]
        new_targets[:, 1:5] = new_bboxes[i]
        new_segments = np.array(new_segments)[i]

    return im, new_targets, new_segments
