# GAN Surgery for Compressed Sensing and Inverse Problems

![GAN Surgery](assets/gan_surgery.png)

# Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

# Datasets and Preprocessing

The 'train' and 'test' split for all models come from the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), at different resolutions.

For BigGAN model, we use [CelebAHQ dataset](https://github.com/tkarras/progressive_growing_of_gans)

The 'out-of-distribution' split comes from the [COCO 2017 Test dataset](http://images.cocodataset.org/zips/test2017.zip) (See [COCO website](http://cocodataset.org/#download) for more details).

Images are preprocessed by center cropping, and saving to tensors for faster training.

A 95:5 train:test split is used.

1. Download the aligned Celeba dataset. This can be done using PyTorch in a python REPL as follows:

Note that the CelebA google drive has limited downloads per day, so if this fails, the contents of the `*.zip` files will be junk, and you must wait and try again.

```python
import torchvision.datasets as d
c = d.CelebA('./data', download=True)
```

2. Now you should have a folder `./data/celeba/img_align_celeba`. Run the preprocessing:

```bash
DATASET=celeba # For DCGAN, use DATASET=celeba64x64
IMG_SIZE=128 # For DCGAN, use IMG_SIZE=64
DATASET_DIR=${DATASET}_preprocessed
python data/preprocess_images.py --dataset $DATASET \
  --input_dir ./data/celeba \
  --output_dir ./data/${DATASET_DIR} \
  --img_size 64 \
  --n -1
```

# Training Generative Models

Before training, you should have run preprocessing as described in [Datasets and Preprocessing](#datasets-and-preprocessing).

## BEGAN

To start training:

```bash
python train_began.py --dataset celeba \
  --dataset_dir ./data/${DATASET_DIR} \
  --batch_size 32 \
  --run_name training \
  --latent_dim 64 \
  --epochs 50 \
  --n_train -1
```

To monitor training:
```bash
tensorboard --logdir ./tensorboard_logs
```

See the [BEGAN model definition](model/began.py) for more details.

## DCGAN
To start training:
```bash
python train_dcgan.py
```

To monitor training:
```bash
tensorboard --logdir ./dcgan_tensorboard_logs
```

See the [DCGAN model definition](model/dcgan.py) for more details.

## VAE and beta-VAE
To train VAE:
```bash
python train_vae.py --epochs 20
```

To train beta-VAE:
```bash
python train_vae.py --epochs 20 --beta 0.1
```

To monitor training:
```bash
tensorboard --logdir ./vae_tensorboard_logs
```
See the [VAE model definition](model/vae.py) for more details.

## BigGAN
BigGAN's model definition is a patched version of the pre-trained model provided by [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BigGAN/).
See the model definition file at [BigGAN](model/biggan.py).

# Running Inverse Imaging Experiments
Our experiments are launched for a fixed set of inverse problems, using fixed optimization hyperparams, for an entire folder of images at a time.

In [settings.py](settings.py), each scenario is described with a keyword, and listed once under `forward_models` (describing the recovery tasks that will be performed), and once under `recovery_settings` (describing the optimization hyperparams that will be used).

To prepare an experiment, copy the desired images to a folder, such as `images/ood-examples`.

For example, performing compressed sensing using BEGAN is described by `began_cs`.
To run the `began_cs` scenario:
```bash
python run_experiments --img_dir ./images/ood-examples --model began_cs
```

To run an experiment and also store Tensorboard logs, use:
```bash
MODEL=began_cs python run_experiments --img_dir ./images/ood-examples --model ${MODEL} --run_dir ${MODEL} --run_name ${MODEL}
```
... and then track progress in tensorboard using:
```bash
tensorboard --logdir ./recovery_tensorboard_logs
```
