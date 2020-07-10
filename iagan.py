import argparse
import os
import shutil
import warnings

import numpy as np

import torch
import torch.nn.functional as F
from forward_model import GaussianCompressiveSensing
from model.began import Generator128
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils import (get_z_vector, load_target_image, load_trained_net,
                   psnr_from_mse)

warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def iagan_recover(
        x,
        gen,
        forward_model,
        optimizer_type='adam',
        mode='clamped_normal',
        limit=1,
        z_lr1=0.1,
        z_lr2=1e-4,
        model_lr=1e-4,
        z_steps1=1600,
        z_steps2=300,
        batch_size=1,
        run_dir=None,  # IAGAN
        run_name=None,  # datetime or config
        set_seed=True,
        disable_tqdm=False,
        return_z1_z2=False,
        **kwargs):

    if set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    z1_dim, z2_dim = gen.input_shapes[0]  # n_cuts = 0

    if (isinstance(forward_model, GaussianCompressiveSensing)):
        n_pixel_bora = 64 * 64 * 3
        n_pixel = np.prod(x.shape)
        noise = torch.randn(batch_size,
                            forward_model.n_measure,
                            device=x.device)
        noise *= 0.1 * torch.sqrt(
            torch.tensor(n_pixel / forward_model.n_measure / n_pixel_bora))

    # z1 is the actual latent code.
    # z2 is the additional input for n_cuts logic (not used here)
    z1 = torch.nn.Parameter(
        get_z_vector((batch_size, *z1_dim),
                     mode=mode,
                     limit=limit,
                     device=x.device))
    params = [z1]
    if len(z2_dim) > 0:
        z2 = torch.nn.Parameter(
            get_z_vector((batch_size, *z2_dim),
                         mode=mode,
                         limit=limit,
                         device=x.device))
        params.append(z2)
    else:
        z2 = None

    if optimizer_type == 'adam':
        optimizer_z = torch.optim.Adam([z1], lr=z_lr1)
        optimizer_model = torch.optim.Adam(gen.parameters(), lr=model_lr)
    else:
        raise NotImplementedError()

    if run_name is not None:
        logdir = os.path.join('recovery_tensorboard_logs', run_dir, run_name)
        if os.path.exists(logdir):
            print("Overwriting pre-existing logs!")
            shutil.rmtree(logdir)
        writer = SummaryWriter(logdir)

    # Save original and distorted image
    if run_name is not None:
        writer.add_image("Original/Clamp", x.clamp(0, 1))
        if forward_model.viewable:
            writer.add_image(
                "Distorted/Clamp",
                forward_model(x.unsqueeze(0).clamp(0, 1)).squeeze(0))

    # Make noisy gaussian measurements
    x = x.expand(batch_size, *x.shape)
    y_observed = forward_model(x)
    if (isinstance(forward_model, GaussianCompressiveSensing)):
        y_observed += noise

    # Stage 1: optimize latent code only
    save_img_every_n = 50
    for j in trange(z_steps1, desc='Stage1', leave=True):
        optimizer_z.zero_grad()
        x_hats = gen.forward(z1, z2, n_cuts=0, **kwargs)
        if gen.rescale:
            x_hats = (x_hats + 1) / 2
        train_mses = F.mse_loss(forward_model(x_hats),
                                y_observed,
                                reduction='none')
        train_mses = train_mses.view(batch_size, -1).mean(1)

        train_mse = train_mses.sum()
        train_mse.backward()
        optimizer_z.step()

        train_mses_clamped = F.mse_loss(forward_model(x_hats.detach().clamp(
            0, 1)),
                                        y_observed,
                                        reduction='none').view(batch_size,
                                                               -1).mean(1)

        orig_mses_clamped = F.mse_loss(x_hats.detach().clamp(0, 1),
                                       x,
                                       reduction='none').view(batch_size,
                                                              -1).mean(1)

        best_train_mse, best_idx = train_mses_clamped.min(0)
        worst_train_mse, worst_idx = train_mses_clamped.max(0)
        best_orig_mse = orig_mses_clamped[best_idx]
        worst_orig_mse = orig_mses_clamped[worst_idx]

        if run_name is not None and j == 0:
            writer.add_image('Start', x_hats[best_idx].clamp(0, 1))

        if run_name is not None:
            writer.add_scalar('Stage1/TRAIN_MSE/best', best_train_mse, j + 1)
            writer.add_scalar('Stage1/TRAIN_MSE/worst', worst_train_mse, j + 1)
            writer.add_scalar('Stage1/TRAIN_MSE/sum', train_mse, j + 1)
            writer.add_scalar('Stage1/ORIG_MSE/best', best_orig_mse, j + 1)
            writer.add_scalar('Stage1/ORIG_MSE/worst', worst_orig_mse, j + 1)
            writer.add_scalar('Stage1/ORIG_PSNR/best',
                              psnr_from_mse(best_orig_mse), j + 1)
            writer.add_scalar('Stage1/ORIG_PSNR/worst',
                              psnr_from_mse(worst_orig_mse), j + 1)

            if j % save_img_every_n == 0:
                writer.add_image('Stage1/Recovered/Best',
                                 x_hats[best_idx].clamp(0, 1), j + 1)

    if run_name is not None:
        writer.add_image('Stage1/Final', x_hats[best_idx].clamp(0, 1))

    # Stage 2: optimize latent code and model
    save_img_every_n = 20
    optimizer_z = torch.optim.Adam([z1], lr=z_lr2)
    for j in trange(z_steps2, desc='Stage2', leave=True):
        optimizer_z.zero_grad()
        optimizer_model.zero_grad()
        x_hats = gen.forward(z1, z2, n_cuts=0, **kwargs)
        if gen.rescale:
            x_hats = (x_hats + 1) / 2
        train_mses = F.mse_loss(forward_model(x_hats),
                                y_observed,
                                reduction='none')
        train_mses = train_mses.view(batch_size, -1).mean(1)

        train_mse = train_mses.sum()
        train_mse.backward()
        optimizer_z.step()
        optimizer_model.step()

        train_mses_clamped = F.mse_loss(forward_model(x_hats.detach().clamp(
            0, 1)),
                                        y_observed,
                                        reduction='none').view(batch_size,
                                                               -1).mean(1)

        orig_mses_clamped = F.mse_loss(x_hats.detach().clamp(0, 1),
                                       x,
                                       reduction='none').view(batch_size,
                                                              -1).mean(1)

        best_train_mse, best_idx = train_mses_clamped.min(0)
        worst_train_mse, worst_idx = train_mses_clamped.max(0)
        best_orig_mse = orig_mses_clamped[best_idx]
        worst_orig_mse = orig_mses_clamped[worst_idx]

        if run_name is not None and j == 0:
            writer.add_image('Start', x_hats[best_idx].clamp(0, 1))

        if run_name is not None:
            writer.add_scalar('Stage2/TRAIN_MSE/best', best_train_mse, j + 1)
            writer.add_scalar('Stage2/TRAIN_MSE/worst', worst_train_mse, j + 1)
            writer.add_scalar('Stage2/TRAIN_MSE/sum', train_mse, j + 1)
            writer.add_scalar('Stage2/ORIG_MSE/best', best_orig_mse, j + 1)
            writer.add_scalar('Stage2/ORIG_MSE/worst', worst_orig_mse, j + 1)
            writer.add_scalar('Stage2/ORIG_PSNR/best',
                              psnr_from_mse(best_orig_mse), j + 1)
            writer.add_scalar('Stage2/ORIG_PSNR/worst',
                              psnr_from_mse(worst_orig_mse), j + 1)

            if j % save_img_every_n == 0:
                writer.add_image('Stage2/Recovered/Best',
                                 x_hats[best_idx].clamp(0, 1), j + 1)

    if run_name is not None:
        writer.add_image('Stage2/Final', x_hats[best_idx].clamp(0, 1))

    if return_z1_z2:
        return x_hats[best_idx], forward_model(x)[0], {'z1': z1, 'z2': z2}
    else:
        return x_hats[best_idx], forward_model(x)[0]


if __name__ == '__main__':
    DEVICE = 'cuda:0'

    a = argparse.ArgumentParser()
    a.add_argument('--img_dir', required=True)
    a.add_argument('--disable_tqdm', default=False)
    args = a.parse_args()

    gen = Generator128(64)
    gen = load_trained_net(
        gen, ('./checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
              '.n_cuts=0/gen_ckpt.49.pt'))
    gen = gen.eval().to(DEVICE)

    img_size = 128
    img_shape = (3, img_size, img_size)

    forward_model = GaussianCompressiveSensing(n_measure=400,
                                               img_shape=img_shape)

    for img_name in tqdm(os.listdir(args.img_dir),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     img_size).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)
        x_hat, x_degraded = iagan_recover(orig_img,
                                          gen,
                                          forward_model,
                                          run_dir='iagan',
                                          run_name=img_basename)