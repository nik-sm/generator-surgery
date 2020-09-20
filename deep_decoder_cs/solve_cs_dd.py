import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from tqdm import tqdm, trange

from deepdecoder.deepdecoder import DeepDecoder

import sys
sys.path.append('..')
from utils import psnr_from_mse

# Command line args:
# - trained or untrained
# - dataset path
# - gamma
# - lbfgs or adam
# - iters

m_ = [12288, 10000, 7500, 5000, 2500, 1000, 750, 500, 400, 300, 200, 100, 50, 30, 20]


def run_cs(args,
           device=('cuda:0' if torch.cuda.is_available() else 'cpu')):

    trans = transforms.Compose([transforms.Resize((args.imsize, args.imsize)), transforms.ToTensor()])

    dataset = datasets.ImageFolder(args.data_path, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    writer = SummaryWriter(f'dd_tensorboard_logs/{datetime.now().isoformat()}')

    for m in args.measures:
        A = torch.randn(args.imsize * args.imsize * 3, m, device=device) * 1 / np.sqrt(m)
        noise = torch.randn(1, m, device=device) * 0.1 / np.sqrt(m)

        for idx, (load_image, _) in enumerate(tqdm(dataloader, desc='Images', leave=True)):
            load_image = load_image.to(device)

            model = DeepDecoder(num_filters=args.num_filters, img_size=args.imsize, depth=args.depth).to(device)
            model.train()

            seed = torch.randn(1, args.num_filters, 4, 4, device=device)
            image = load_image.view([1, -1])

            opt = torch.optim.Adam(model.parameters(), lr=args.lr)

            y = torch.matmul(image, A) + noise

            def closure():
                opt.zero_grad()
                rep = model.forward(seed).view([1, -1])
                rep_y = torch.matmul(rep, A)
                loss = ((rep_y - y)**2).sum(dim=1).mean()
                loss.backward()
                return loss

            for step in trange(args.iters, desc='fit', leave=False):
                loss = opt.step(closure)
                writer.add_scalar(f'{idx}/TRAIN_MSE', loss, step)
                writer.add_scalar(f'{idx}/TRAIN_PSNR', psnr_from_mse(loss), step)
                with torch.no_grad():
                    rec = model(seed)
                    orig_mse = F.mse_loss(rec, load_image)
                    writer.add_scalar(f'{idx}/ORIG_MSE', orig_mse, step)
                    writer.add_scalar(f'{idx}/ORIG_PSNR', psnr_from_mse(orig_mse), step)

            writer.add_image(f'{idx}/Final', rec)
            del seed, opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS with Deep Decoder architecture")
    parser.add_argument("--num_filters", type=int, default=64, help=" TODO Number of filters in each Deep Decoder layer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test set images")
    parser.add_argument("--lr", type=float, help="Learning rate for optimizer", default=0.01)
    parser.add_argument("--iters", type=int, help="Number of inversion iterations", default=1900)
    parser.add_argument("--depth", type=int, help="depth of deep decoder", default=5)
    parser.add_argument("--cpu", help="Use CPU rather than CUDA", action="store_true")
    parser.add_argument("--name",
                        help="Name of these experiments, to be used in saving artifacts",
                        type=str,
                        default="artifacts")
    parser.add_argument("--imsize", type=int, required=True, help="Size of images to use")
    parser.add_argument('--measures', type=int, nargs='+', help='no. of measurements', default=m_)

    args = parser.parse_args()

    run_cs(args)
