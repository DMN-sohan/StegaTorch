import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stn import spatial_transformer_network as stn_transformer
import utils

class StegaStampEncoder(nn.Module):
    def __init__(self, height, width):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = nn.Linear(100, 7500)

        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.up8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.up9 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1)

    def forward(self, secret, image):
        secret = secret - 0.5
        image = image - 0.5

        secret = F.relu(self.secret_dense(secret))
        secret = secret.view(-1, 3, 50, 50)
        secret_enlarged = F.interpolate(secret, size=(400, 400), mode='nearest')

        inputs = torch.cat([secret_enlarged, image], dim=1)
        conv1 = F.relu(self.conv1(inputs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        up6 = F.relu(self.up6(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))
        up7 = F.relu(self.up7(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))
        up8 = F.relu(self.up8(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))
        up9 = F.relu(self.up9(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = F.relu(self.conv9(merge9))
        conv10 = F.relu(self.conv10(conv9))
        residual = self.residual(conv10)
        return residual

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size, height, width):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (height // 8) * (width // 8), 128),
            nn.ReLU()
        )
        initial = np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten()
        self.W_fc1 = nn.Parameter(torch.zeros(128, 6))
        self.b_fc1 = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (height // 64) * (width // 64), 512),
            nn.ReLU(),
            nn.Linear(512, secret_size)
        )

    def forward(self, image):
        image = image - 0.5
        stn_params = self.stn_params(image)
        x = torch.matmul(stn_params, self.W_fc1) + self.b_fc1
        transformed_image = stn_transformer(image, x, [self.height, self.width, 3])
        return self.decoder(transformed_image)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, image):
        x = image - 0.5
        x = self.model(x)
        output = torch.mean(x)
        return output, x

def transform_net(encoded_image, args, global_step):
    sh = encoded_image.shape

    ramp_fn = lambda ramp: min(float(global_step) / ramp, 1.)

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_tf(rnd_bri, rnd_hue, args.batch_size)

    jpeg_quality = 100. - torch.rand(1).item() * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
    jpeg_factor = 5000. / jpeg_quality if jpeg_quality < 50 else 200. - jpeg_quality * 2 / 100. + 0.0001

    rnd_noise = torch.rand(1).item() * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1).item() * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # blur
    f = utils.random_blur_kernel(probs=[.25, .25], N_blur=7,
                                 sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
    encoded_image = F.conv2d(encoded_image, f, padding=3)

    noise = torch.randn_like(encoded_image) * rnd_noise
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    contrast_scale = torch.rand(encoded_image.shape[0], 1, 1, 1, device=encoded_image.device) * \
                     (contrast_params[1] - contrast_params[0]) + contrast_params[0]

    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    encoded_image = utils.random_saturation_tf(encoded_image, rnd_sat)
    encoded_image = torch.clamp(encoded_image, 0, 1)

    encoded_image = torch.clamp(encoded_image, 0, 1)

    encoded_image_jpeg = utils.jpeg_compress_decompress(encoded_image, rounding_approx=args.rounding_approx,
                                                        factor=jpeg_factor)
    return encoded_image_jpeg, encoded_image
