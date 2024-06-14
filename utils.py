import cv2
import itertools
import numpy as np
import random
import torch
import torch.nn.functional as F


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = torch.stack(torch.meshgrid(torch.arange(N_blur), torch.arange(N_blur), indexing='ij'), -1).float() - (.5 * (N - 1))
    manhat = torch.sum(torch.abs(coords), -1)

    # nothing, default
    vals_nothing = (manhat < .5).float()

    # gauss
    sig_gauss = torch.empty(1).uniform_(sigrange_gauss[0], sigrange_gauss[1]).item()
    vals_gauss = torch.exp(-torch.sum(coords**2, -1) / 2. / sig_gauss**2)

    # line
    theta = torch.empty(1).uniform_(0, 2. * np.pi).item()
    v = torch.tensor([torch.cos(theta), torch.sin(theta)])
    dists = torch.sum(coords * v, -1)

    sig_line = torch.empty(1).uniform_(sigrange_line[0], sigrange_line[1]).item()
    w_line = torch.empty(1).uniform_(wmin_line, .5 * (N - 1) + .1).item()

    vals_line = torch.exp(-dists**2 / 2. / sig_line**2) * (manhat < w_line).float()

    t = torch.empty(1).uniform_().item()
    vals = vals_nothing
    if t < probs[0] + probs[1]:
        vals = vals_line
    if t < probs[0]:
        vals = vals_gauss

    v = vals / torch.sum(vals)
    z = torch.zeros_like(v)
    f = torch.reshape(torch.stack([v, z, z, z, v, z, z, z, v], -1), [N, N, 3, 3])

    return f


def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 8))

    for i in range(batch_size):
        tl_x = random.uniform(-d, d)     # Top left corner, top
        tl_y = random.uniform(-d, d)    # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)    # Bot left corner, left
        tr_x = random.uniform(-d, d)     # Top right corner, top
        tr_y = random.uniform(-d, d)   # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)   # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y +  image_size]], dtype="float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i, 0, :] = M_inv.flatten()[:8]
        Ms[i, 1, :] = M.flatten()[:8]
    return Ms


def get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size):
    rnd_hue = torch.empty((batch_size, 1, 1, 3)).uniform_(-rnd_hue, rnd_hue)
    rnd_brightness = torch.empty((batch_size, 1, 1, 1)).uniform_(-rnd_bri, rnd_bri)
    return rnd_hue + rnd_brightness


## Differentiable JPEG, Source - https://github.com/rshin/differentiable-jpeg/blob/master/jpeg-tensorflow.ipynb

# 1. RGB -> YCbCr
# https://en.wikipedia.org/wiki/YCbCr
def rgb_to_ycbcr(image):
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.],
         [112., -93.786, -18.214]],
        dtype=np.float32).T / 255
    shift = [16., 128., 128.]

    result = torch.tensordot(image, torch.tensor(matrix), dims=1) + torch.tensor(shift)
    return result


def rgb_to_ycbcr_jpeg(image):
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=np.float32).T
    shift = [0., 128., 128.]

    result = torch.tensordot(image, torch.tensor(matrix), dims=1) + torch.tensor(shift)
    return result


# 2. Chroma subsampling
def downsampling_420(image):
    # input: batch x height x width x 3
    # output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    y, cb, cr = torch.split(image, 1, dim=3)
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2, padding=0)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2, padding=0)
    return y.squeeze(-1), cb.squeeze(-1), cr.squeeze(-1)


# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def image_to_patches(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    k = 8
    batch_size, height, width = image.shape
    image_reshaped = image.view(batch_size, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, -1, k, k)


# 4. DCT
def dct_8x8_ref(image):
    image = image - 128
    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        result[u, v] = value
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    return result * scale


def dct_8x8(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = scale * torch.tensordot(image, torch.tensor(tensor), dims=2)
    return result


# 5. Quantization
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
c_table = np.array(
    [[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99,
                                        99], [24, 26, 56, 99, 99, 99, 99, 99],
     [47, 66, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99,
                                        99], [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]],
    dtype=np.float32).T


def quality_to_factor(quality):
    if quality < 50:
        factor = 5000. / quality
    else:
        factor = 200. - quality * 2
    return factor / 100.


def quantize(image, factor, table):
    result = torch.round(image / (torch.tensor(table) * factor))
    return result


# 6. Dequantization
def dequantize(image, factor, table):
    result = image * (torch.tensor(table) * factor)
    return result


# 7. Inverse DCT
def idct_8x8_ref(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    image = image * scale

    result = np.zeros((8, 8), dtype=np.float32)
    for x, y in itertools.product(range(8), range(8)):
        value = 0
        for u, v in itertools.product(range(8), range(8)):
            value += image[u, v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        result[x, y] = value
    return result


def idct_8x8(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    image = image * scale

    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    result = torch.tensordot(image, torch.tensor(tensor), dims=2)
    return result


# 8. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = patches.shape[0]
    image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, height, width)


# 9. Upsampling
def upsampling_420(y, cb, cr):
    # input: batch x height x width, batch x height/2 x width/2, batch x height/2 x width/2
    # output: batch x height x width x 3
    cb = F.interpolate(cb.unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1)
    cr = F.interpolate(cr.unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1)
    return torch.stack([y, cb, cr], dim=3)


# 10. YCbCr -> RGB
def ycbcr_to_rgb(image):
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.],
         [112., -93.786, -18.214]],
        dtype=np.float32).T / 255
    shift = [16., 128., 128.]

    matrix_inv = np.linalg.inv(matrix)
    shift_inv = -np.dot(shift, matrix_inv)

    result = torch.tensordot(image, torch.tensor(matrix_inv), dims=1) + torch.tensor(shift_inv)
    return result


def ycbcr_to_rgb_jpeg(image):
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=np.float32).T

    matrix_inv = np.linalg.inv(matrix)
    shift = [0., 128., 128.]
    shift_inv = -np.dot(shift, matrix_inv)

    result = torch.tensordot(image, torch.tensor(matrix_inv), dims=1) + torch.tensor(shift_inv)
    return result


# Full Jpeg

def diff_round(x):
    return torch.round(x) + (x - torch.round(x))**3


def jpeg_compress_decompress(image, downsampling=True):
    # image = torch.tensor(image)

    image = rgb_to_ycbcr_jpeg(image)

    if downsampling:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = torch.split(image, 1, dim=3)
        y, cb, cr = y.squeeze(-1), cb.squeeze(-1), cr.squeeze(-1)

    components = {'y': y, 'cb': cb, 'cr': cr}
    factor = quality_to_factor(75)

    for k in components.keys():
        comp = components[k]
        table = c_table if k in ('cb', 'cr') else y_table

        shape = comp.shape

        patches = image_to_patches(comp)
        patches = patches.view(-1, 8, 8)

        dct_patches = torch.stack([dct_8x8(patch) for patch in patches])
        quant_patches = torch.stack(
            [quantize(patch, factor, table) for patch in dct_patches])

        dequant_patches = torch.stack(
            [dequantize(patch, factor, table) for patch in quant_patches])
        idct_patches = torch.stack(
            [idct_8x8(patch) for patch in dequant_patches])

        idct_patches = idct_patches.view(shape[0], -1, 8, 8)
        comp = patches_to_image(idct_patches, shape[1], shape[2])

        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']

    if downsampling:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.stack([y, cb, cr], dim=3)

    image = ycbcr_to_rgb_jpeg(image)
    image = torch.clip(image, 0, 255)
    return image

