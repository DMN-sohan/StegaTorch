import glob
import os
from PIL import Image, ImageOps
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import models
from os.path import join
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

TRAIN_PATH = './data/mirflickr/images1/images/'
LOGS_PATH = "./logs/"
CHECKPOINTS_PATH = './checkpoints/'
SAVED_MODELS = './saved_models'

if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, files_list, secret_size, size=(400, 400)):
        self.files_list = files_list
        self.secret_size = secret_size
        self.size = size

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]
        try:
            img_cover = Image.open(img_cover_path).convert("RGB")
            img_cover = ImageOps.fit(img_cover, self.size)
            img_cover = np.array(img_cover, dtype=np.float32) / 255.
        except:
            img_cover = np.zeros((self.size[0], self.size[1], 3), dtype=np.float32)

        secret = np.random.binomial(1, .5, self.secret_size).astype(np.float32)
        return img_cover, secret

def get_img_batch(files_list, secret_size, batch_size=4, size=(400, 400)):
    dataset = ImageDataset(files_list, secret_size, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return next(iter(dataloader))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--secret_size', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=140000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--l2_loss_scale', type=float, default=1.5)
    parser.add_argument('--l2_loss_ramp', type=int, default=20000)
    parser.add_argument('--l2_edge_gain', type=float, default=10.0)
    parser.add_argument('--l2_edge_ramp', type=int, default=20000)
    parser.add_argument('--l2_edge_delay', type=int, default=60000)
    parser.add_argument('--lpips_loss_scale', type=float, default=1)
    parser.add_argument('--lpips_loss_ramp', type=int, default=20000)
    parser.add_argument('--secret_loss_scale', type=float, default=1)
    parser.add_argument('--secret_loss_ramp', type=int, default=1)
    parser.add_argument('--G_loss_scale', type=float, default=1)
    parser.add_argument('--G_loss_ramp', type=int, default=20000)
    parser.add_argument('--borders', type=str, choices=['no_edge', 'black', 'random', 'randomrgb', 'image', 'white'], default='black')
    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=1.0)
    parser.add_argument('--v_scale', type=float, default=1.0)
    parser.add_argument('--no_gan', action='store_true')
    parser.add_argument('--rnd_trans', type=float, default=.1)
    parser.add_argument('--rnd_bri', type=float, default=.3)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--rnd_sat', type=float, default=1.0)
    parser.add_argument('--rnd_hue', type=float, default=.1)
    parser.add_argument('--contrast_low', type=float, default=.5)
    parser.add_argument('--contrast_high', type=float, default=1.5)
    parser.add_argument('--jpeg_quality', type=float, default=25)
    parser.add_argument('--no_jpeg', action='store_true')
    parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
    parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
    parser.add_argument('--contrast_ramp', type=int, default=1000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=1000)
    parser.add_argument('--no_im_loss_steps', help="Train without image loss for first x steps", type=int, default=500)
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    EXP_NAME = args.exp_name

    files_list = glob.glob(join(TRAIN_PATH, "**/*"))

    height = 400
    width = 400

    writer = SummaryWriter(join(LOGS_PATH, EXP_NAME))

    encoder = models.StegaStampEncoder(height=height, width=width).to(device)
    decoder = models.StegaStampDecoder(secret_size=args.secret_size, height=height, width=width).to(device)
    discriminator = models.Discriminator().to(device)

    optimizer_G = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=.00001)

    criterion = nn.MSELoss()

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        discriminator.load_state_dict(checkpoint['discriminator'])

    total_steps = len(files_list) // args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            no_im_loss = global_step < args.no_im_loss_steps
            images, secrets = get_img_batch(files_list=files_list, secret_size=args.secret_size, batch_size=args.batch_size, size=(height, width))
            images = torch.tensor(images).to(device)
            secrets = torch.tensor(secrets).to(device)

            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp, args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp, args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran
            M = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)

            M = torch.tensor(M).to(device)

            encoder.train()
            decoder.train()

            optimizer_G.zero_grad()
            if no_im_loss:
                encoded_img = encoder(images, secrets)
                decoded_secret = decoder(encoded_img)
                secret_loss = criterion(decoded_secret, secrets)
                secret_loss.backward()
                optimizer_G.step()
            else:
                encoded_img = encoder(images, secrets)
                decoded_secret = decoder(encoded_img)
                secret_loss = criterion(decoded_secret, secrets)

                real_output = discriminator(images)
                fake_output = discriminator(encoded_img.detach())
                D_loss = criterion(real_output, torch.ones_like(real_output).to(device)) + \
                         criterion(fake_output, torch.zeros_like(fake_output).to(device))

                G_loss = criterion(fake_output, torch.ones_like(fake_output).to(device))

                loss = secret_loss + l2_loss_scale * G_loss + lpips_loss_scale * D_loss
                loss.backward()
                optimizer_G.step()
                if not args.no_gan:
                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()

            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar('Loss/secret_loss', secret_loss.item(), global_step)
                writer.add_scalar('Loss/D_loss', D_loss.item(), global_step)
                writer.add_scalar('Loss/G_loss', G_loss.item(), global_step)

            if global_step % 10000 == 0:
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'discriminator': discriminator.state_dict(),
                }, join(CHECKPOINTS_PATH, EXP_NAME, f'{EXP_NAME}_step{global_step}.pth'))

if __name__ == "__main__":
    main()
