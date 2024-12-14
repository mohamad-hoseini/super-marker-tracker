
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

import pose_helpers as pose_helpers


def randomize_bbox_coordinates(bbox, im_shape, random_factor_range=(0.1, 0.3)):
    bbox = np.array(bbox)
    x1, x2, y1, y2 = bbox
    x_range = x2 - x1
    y_range = y2 - y1
    x_min = int(x1 - x_range * get_random_factor(random_factor_range))
    x_max = int(x2 + x_range * get_random_factor(random_factor_range))
    y_min = int(y1 - y_range * get_random_factor(random_factor_range))
    y_max = int(y2 + y_range * get_random_factor(random_factor_range))
    x_min = max(0, x_min)
    x_max = min(im_shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(im_shape[1], y_max)
    bbox = np.array([x_min, x_max, y_min, y_max])
    return bbox


def get_random_factor(factor_range):

    factor = np.random.uniform(factor_range[0], factor_range[1])
    return factor


def get_cropped_imgs(imgs, bbox):
    x1, x2, y1, y2 = (np.round(bbox)).astype(int)
    batch_size = imgs.shape[0]
    nchannels = imgs.shape[1]
    cropped_imgs = np.empty((batch_size, nchannels, x2 - x1, y2 - y1))
    for i in range(batch_size):
        for n in range(nchannels):
            cropped_imgs[i, n] = imgs[i, n, x1:x2, y1:y2]
    return cropped_imgs


def pad_keypoints(keypoints, pad_h, pad_w):
    keypoints[:, 0] += pad_h
    keypoints[:, 1] += pad_w
    return keypoints


def pad_img_to_square(img, bbox=None):
    if bbox is not None:  # Check if bbox is square
        x1, x2, y1, y2 = bbox
        dx, dy = x2 - x1, y2 - y1
    else:
        dx, dy = img.shape[-2:]

    if dx == dy:
        return img, (0, 0, 0, 0)

    largest_dim = max(dx, dy)
    if (dx < largest_dim and abs(dx - largest_dim) % 2 != 0) or (
        dy < largest_dim and abs(dy - largest_dim) % 2 != 0
    ):
        largest_dim += 1

    if dx < largest_dim:
        pad_x = abs(dx - largest_dim)
        pad_x_left = pad_x // 2
        pad_x_right = pad_x - pad_x_left
    else:
        pad_x_left = 0
        pad_x_right = 0

    if dy < largest_dim:
        pad_y = abs(dy - largest_dim)
        pad_y_top = pad_y // 2
        pad_y_bottom = pad_y - pad_y_top
    else:
        pad_y_top = 0
        pad_y_bottom = 0

    if img.ndim > 3:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, 0, 0, 0, 0)
    elif img.ndim == 3:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, 0, 0)
    else:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)

    img = F.pad(
        img,
        pads,
        mode="constant",
        value=0,
    )

    return img, (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)


def resize_keypoints(keypoints, desired_shape, original_shape):
    x_scale = desired_shape[1] / original_shape[1]  # scale factor for x coordinates
    y_scale = desired_shape[0] / original_shape[0]  # scale factor for y coordinates
    xlabels, ylabels = keypoints[:, 0], keypoints[:, 1]
    xlabels = xlabels * x_scale
    ylabels = ylabels * y_scale
    # Stack the x and y coordinates together using torch.stack
    keypoints = torch.stack([xlabels, ylabels], dim=1)
    return keypoints


def resize_image(im, resize_shape):
    h, w = resize_shape
    if im.ndim == 3:
        im = torch.unsqueeze(im, dim=0)
    elif im.ndim == 2:
        im = torch.unsqueeze(im, dim=0)
        im = torch.unsqueeze(im, dim=0)
    im = F.interpolate(im, size=(h, w), mode="bilinear", align_corners=True).squeeze(
        dim=0
    )
    return im


def get_crop_resize_params(img, x_dims, y_dims, xy=(256, 256)):
    x1 = int(x_dims[0])
    x2 = int(x_dims[1])
    y1 = int(y_dims[0])
    y2 = int(y_dims[1])

    resize = False
    if abs(y2 - y1) > xy[0]:  # if cropped image larger than desired size
        # crop image then resize image and landmarks
        resize = True
    else:  # if cropped image smaller than desired size then add padding accounting for labels in view
        y_pad = abs(abs(y2 - y1) - xy[0])
        if y_pad % 2 == 0:
            y_pad = y_pad // 2
            y1, y2 = y1 - y_pad, y2 + y_pad
        else:  # odd number division so add 1
            y_pad = y_pad // 2
            y1, y2 = y1 - y_pad, y2 + y_pad + 1

    if abs(x2 - x1) > xy[1]:  # if cropped image larger than desired size
        resize = True
    else:
        x_pad = abs(abs(x2 - x1) - xy[1])
        if x_pad % 2 == 0:
            x_pad = x_pad // 2
            x1, x2 = x1 - x_pad, x2 + x_pad
        else:
            x_pad = x_pad // 2
            x1, x2 = x1 - x_pad, x2 + x_pad + 1

    if y2 > img.shape[1]:
        y1 -= y2 - img.shape[1]
    if x2 > img.shape[0]:
        x1 -= x2 - img.shape[0]

    y2, x2 = min(y2, img.shape[1]), min(x2, img.shape[0])
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = max(y2, xy[0]), max(x2, xy[1])

    return x1, x2, y1, y2, resize


def crop_image(im, bbox=None):
    if bbox is None:
        return im
    y1, y2, x1, x2 = bbox
    if im.ndim == 2:
        im = im[y1:y2, x1:x2]
    elif im.ndim == 3:
        im = im[:, y1:y2, x1:x2]
    elif im.ndim == 4:
        im = im[:, :, y1:y2, x1:x2]
    else:
        raise ValueError("Cannot handle image with ndim=" + str(im.ndim))
    return im


def adjust_keypoints(xlabels, ylabels, crop_xy, padding, current_size, desired_size):
    # Rescale keypoints to original image size
    xlabels, ylabels = rescale_keypoints(xlabels, ylabels, current_size, desired_size)
    xlabels, ylabels = adjust_keypoints_for_padding(xlabels, ylabels, padding)
    # Adjust for cropping
    x1, y1 = crop_xy[0], crop_xy[1]
    xlabels += x1
    ylabels += y1
    return xlabels, ylabels


def rescale_keypoints(xlabels, ylabels, current_size, desired_size):
    xlabels *= desired_size[1] / current_size[1]  # x_scale
    ylabels *= desired_size[0] / current_size[0]  # y_scale
    return xlabels, ylabels


def adjust_keypoints_for_padding(xlabels, ylabels, pads):
    pad_y_top, pad_y_bottom, pad_x_left, pad_x_right = pads
    xlabels -= pad_y_top
    ylabels -= pad_x_left
    return xlabels, ylabels


def adjust_bbox(prev_bbox, img_yx, div=16, extra=1):
    x1, x2, y1, y2 = np.round(prev_bbox)
    xdim, ydim = (x2 - x1), (y2 - y1)

    # Pad bbox dimensions to be divisible by div
    Lpad = int(div * np.ceil(xdim / div) - xdim)
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    Lpad = int(div * np.ceil(ydim / div) - ydim)
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2

    x1, x2, y1, y2 = x1 - xpad1, x2 + xpad2, y1 - ypad1, y2 + ypad2
    xdim = min(x2 - x1, img_yx[1])
    ydim = min(y2 - y1, img_yx[0])

    # Choose largest dimension for image size
    if xdim > ydim:
        # Adjust ydim
        ypad = xdim - ydim
        if ypad % 2 != 0:
            ypad += 1
        y1 = max(0, y1 - ypad // 2)
        y2 = min(y2 + ypad // 2, img_yx[0])
    else:
        # Adjust xdim
        xpad = ydim - xdim
        if xpad % 2 != 0:
            xpad += 1
        x1 = max(0, x1 - xpad // 2)
        x2 = min(x2 + xpad // 2, img_yx[1])
    adjusted_bbox = (x1, x2, y1, y2)
    return adjusted_bbox


def augment_data(
    image,
    keypoints,
    scale=False,
    scale_range=0.5,
    rotation=False,
    rotation_range=10,
    flip=True,
    contrast_adjust=True,
):
    if scale:
        scale_range = max(0, min(2, float(scale_range)))
        scale_factor = (np.random.rand() - 0.5) * scale_range + 1
        image = image.squeeze() * scale_factor
        keypoints = keypoints * scale_factor
    if rotation:
        theta = np.random.rand() * rotation_range - rotation_range / 2
        print("rotating by {}".format(theta))
        image = ndimage.rotate(image, theta, axes=(-2, -1), reshape=True)
        keypoints = rotate_points(keypoints, theta)  # TODO: Add rotation function
    if flip and np.random.rand() > 0.5:
        keypoints[:, 0] = 256 - keypoints[:, 0]
        image = ndimage.rotate(image, 180, axes=(-1, 0), reshape=True)
    if contrast_adjust and np.random.rand() > 0.5:
        image = pose_helpers.randomly_adjust_contrast(image)

    return image, keypoints
