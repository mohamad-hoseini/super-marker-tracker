
import random
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

N_FACTOR = 2**4 // (2**2)
SIGMA = 3 * 4 / N_FACTOR
Lx = 64

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def predict(net, im_input, smooth=False):
    lx = int(net.image_shape[0] / N_FACTOR)
    ly = int(net.image_shape[1] / N_FACTOR)
    batch_size = im_input.shape[0]
    num_keypoints = 15
    locx_mesh, locy_mesh = torch.meshgrid(
        torch.arange(batch_size), torch.arange(num_keypoints), indexing="ij"
    )
    locx_mesh = locx_mesh.to(net.device)
    locy_mesh = locy_mesh.to(net.device)

    # Predict
    with torch.no_grad():
        hm_pred, locx_pred, locy_pred = net(im_input)

        if smooth:
            hm_pred = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])

        hm_pred = hm_pred.reshape(batch_size, num_keypoints, lx * ly)
        locx_pred = locx_pred.reshape(batch_size, num_keypoints, lx * ly)
        locy_pred = locy_pred.reshape(batch_size, num_keypoints, lx * ly)

        # likelihood, imax = torch.max(hm_pred, -1)
        _, imax = torch.max(hm_pred, -1)
        likelihood = torch.sigmoid(hm_pred)
        likelihood, _ = torch.max(likelihood, -1)
        i_y = imax % lx
        i_x = torch.div(imax, lx, rounding_mode="trunc")
        x_pred = (locx_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_x
        y_pred = (locy_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_y

    x_pred *= N_FACTOR
    y_pred *= N_FACTOR

    return y_pred, x_pred, likelihood


def numpy_predict(net, im_input, smooth=False):
    lx = int(net.image_shape[0] / N_FACTOR)
    ly = int(net.image_shape[1] / N_FACTOR)
    batch_size = im_input.shape[0]
    num_keypoints = len(net.bodylabels)
    locx_mesh, locy_mesh = np.meshgrid(
        np.arange(batch_size), np.arange(num_keypoints), indexing="ij"
    )

    # Predict
    with torch.no_grad():
        hm_pred, locx_pred, locy_pred = net(im_input)

        if smooth:
            hm_pred = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])

        hm_pred = hm_pred.reshape(batch_size, num_keypoints, lx * ly).cpu().numpy()
        locx_pred = locx_pred.reshape(batch_size, num_keypoints, lx * ly).cpu().numpy()
        locy_pred = locy_pred.reshape(batch_size, num_keypoints, lx * ly).cpu().numpy()

        likelihood = np.maximum(hm_pred, -1)
        imax = np.argmax(hm_pred, -1)
        i_y = imax % lx
        i_x = imax // lx
        x_pred = (locx_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_x
        y_pred = (locy_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_y

    x_pred *= N_FACTOR
    y_pred *= N_FACTOR

    return y_pred, x_pred, likelihood


def randomly_adjust_contrast(img):
    """
    Randomly adjusts contrast of image
    img: ND-array of size nchan x Ly x Lx
    Assumes image values in range 0 to 1
    """
    brange = [-0.2, 0.2]
    bdiff = brange[1] - brange[0]
    crange = [0.7, 1.3]
    cdiff = crange[1] - crange[0]
    imax = img.max()
    if (bdiff < 0.01) and (cdiff < 0.01):
        return img
    bfactor = np.random.rand() * bdiff + brange[0]
    cfactor = np.random.rand() * cdiff + crange[0]
    mm = img.mean()
    jj = img + bfactor * imax
    jj = np.minimum(imax, (jj - mm) * cfactor + mm)
    jj = jj.clip(0, imax)
    return jj

