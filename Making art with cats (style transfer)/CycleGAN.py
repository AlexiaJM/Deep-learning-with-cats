# Reference 1 : https://arxiv.org/pdf/1703.10593.pdf
# Reference 2 : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/DCGAN
# Uses spaces instead of tabs, as opposed to my other python files

#### Example after training:
# python CycleGAN.py --trained_model True --Path_load /home/user/Output/CycleGAN/run-5/models/X_epoch_200.pth --image_size 700 --resize_type "crop"

#### Important Note
# So far very poor results after 100 epochs and often contain artifacts. Also very slow, takes about 10 hours for 100 epochs.
# It's way too slow to experiment much and be able to optimize the hyperparameters.
# This idea will probably be abandoned in favour of fast neural transfer.

## Parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--resize_type', default='scale_crop', help='If resize_type is not "" we resize the images. Only a trained model can decide not to resize and be sure to have a lot of CPU/GPY memory! "crop" only use a center crop (To be used only after training) and "scale" only rescale and "scale_crop" does a rescale and random crop.')
parser.add_argument('--image_size_before_crop', type=int, default=286, help='if larger than image_size, we resize the image to "image_size_before_crop" and then randomly crop to make it "image_size". This adds random jitter.') #CycleGAN original value
parser.add_argument('--image_size', type=int, default=256, help='Final image size') #CycleGAN original value
parser.add_argument('--batch_size', type=int, default=1) #CycleGAN original value
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--G_h_size', type=int, default=64, help='Number of hidden nodes in the Generator. Too small leads to bad results, too big blows up the GPU RAM.') #CycleGAN original value
parser.add_argument('--D_h_size', type=int, default=64, help='Number of hidden nodes in the Discriminator. Too small leads to bad results, too big blows up the GPU RAM.') #CycleGAN original value
parser.add_argument('--G_residual_blocks', type=int, default=9, help='The authors used 6 blocks for 128x128 images and 9 blocks for 256x256 or higher images.') #CycleGAN original value
parser.add_argument('--pool_size', type=int, default=50, help='Size of fake image pool') #CycleGAN original value
parser.add_argument('--lr', type=float, default=.0002, help='Learning rate')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of normal epochs')
parser.add_argument('--n_epoch_decay', type=int, default=100, help='Number of decaying epochs')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0]') #CycleGAN original value
parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]') #CycleGAN original value
parser.add_argument('--penalty', type=float, default=10, help='Cycle consistency penalty parameter') #CycleGAN original value
parser.add_argument('--identity_penalty', type=float, default=5, help='Identity loss penalty parameter, important to prevent change in colors when doing painting <-> photo.')
parser.add_argument('--image_pool_size', type=int, default=50, help='Cycle consistency penalty parameter') #CycleGAN original value
parser.add_argument('--norm_type', default='instance', help='If "instance" uses instance normalization; if "batch" uses batch normalization.')
parser.add_argument('--use_dropout', type=bool, default=False, help='If True, uses dropout in the residual blocks') #CycleGAN original value
parser.add_argument('--no_flip', type=bool, default=False, help='If True, do not flip') #CycleGAN original value
parser.add_argument('--seed', type=int)
parser.add_argument('--input1_folder', default='/home/alexia/Datasets/Meow', help='Domain 1 images (cats) folder')
parser.add_argument('--input2_folder', default='/home/alexia/Datasets/ukiyoe', help='Domain 2 images (for style) folder')
parser.add_argument('--output_folder', default='/media/alexia/9aa6f061-db8b-48fa-9559-eba69dcebaa1/Output/CycleGAN', help='output folder')
parser.add_argument('--Path_load', default='', help='Full path to models to load with a X in place of G1, G2, D1, D2 (Must be like this: /home/output_folder/run-5/models/X_epoch_11.pth will grab G1_epoch_11.pth, G2_epoch_11.pth, D1_epoch_11.pth and D2_epoch_11.pth)')
parser.add_argument('--trained_model', type=bool, default=False, help='If True, the model has been trained and we only want it to generate the pictures (not resized).')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--n_workers', type=int, default=2, help='Number of subprocess to use to load the data. Use at least 2 or the number of cpu cores - 1.')
param = parser.parse_args()

## Imports

# Time
import time
start = time.time()

# Check folder run-i for all i=0,1,... until it finds run-j which does not exists, then creates a new folder run-j
import os
run = 0
base_dir = f"{param.output_folder}/run-{run}"
while os.path.exists(base_dir):
    run += 1
    base_dir = f"{param.output_folder}/run-{run}"
os.mkdir(base_dir)
logs_dir = f"{base_dir}/logs"
os.mkdir(logs_dir)
os.mkdir(f"{base_dir}/images")
os.mkdir(f"{base_dir}/models")

# where we save the output
log_output = open(f"{logs_dir}/log.txt", 'w')
print(param)
print(param, file=log_output)

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn

# For plotting the Loss of D and G using tensorboard
from tensorboard_logger import configure, log_value
configure(logs_dir, flush_secs=5)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import PIL

if param.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

# To see images
from IPython.display import Image
to_img = transf.ToPILImage()

import math
import itertools
import random
import numpy as np

## Setting seed
import random
param.seed = param.seed or random.randint(1, 10000)
print(f"Random Seed: {param.seed}")
print(f"Random Seed: {param.seed}", file=log_output)
random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
    torch.cuda.manual_seed_all(param.seed)

## Transforming images
# CycleGAN use PIL.Image.BICUBIC while TorchVision default is PIL.Image.BILINEAR
trans = []
if param.resize_type == "crop":
    trans.append(transf.CenterCrop((param.image_size, param.image_size)))
elif param.resize_type == "resize":
    trans.append(transf.Scale((param.image_size, param.image_size), PIL.Image.BICUBIC))
elif param.resize_type == "crop_resize":
    trans.append(transf.Scale((param.image_size_before_crop, param.image_size_before_crop), PIL.Image.BICUBIC))
    trans.append(transf.RandomCrop(param.image_size))
if not param.trained_model:
    if not param.no_flip:
        trans.append(transf.RandomHorizontalFlip())
trans.append(transf.ToTensor())
trans.append(transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]))
trans = transf.Compose(trans)

## Importing datasets
class MultipleDatasets(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
    def __len__(self):
        return min(len(d) for d in self.datasets)

data1 = dset.ImageFolder(root=param.input1_folder, transform=trans)
data2 = dset.ImageFolder(root=param.input2_folder, transform=trans)
data = MultipleDatasets(data1, data2)

# Loading data in batch
if param.trained_model:
    param.batch_size = 1
dataset = torch.utils.data.DataLoader(data, batch_size=param.batch_size, shuffle=True, num_workers=param.n_workers, drop_last=True)
# One epoch is trough the minimum sample size of one domain

# Set batch norm or instance norm
import functools
if param.norm_type == 'batch':
    Norm2D = functools.partial(nn.BatchNorm2d)
elif param.norm_type == 'instance':
    Norm2D = functools.partial(nn.InstanceNorm2d)

## Models

# Residual Block of Generator
class Residual_block(torch.nn.Module):
    def __init__(self, h_size):
        super(Residual_block, self).__init__()
        # Two Conv layers with same output size
        model = [nn.ReflectionPad2d(padding=1),
                 nn.Conv2d(h_size, h_size, kernel_size=3, stride=1, padding=0),
                 Norm2D(h_size),
                 nn.ReLU(True)]
        if param.use_dropout:
            model += [nn.Dropout(0.5)]
        model += [nn.ReflectionPad2d(padding=1),
                 nn.Conv2d(h_size, h_size, kernel_size=3, stride=1, padding=0),
                 nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # Return itself + the result of the two convolutions
        # Can't do this in parallel
        output = self.model(input) + input
        return output

# Generator
class CycleGAN_G(torch.nn.Module):
    def __init__(self):
        super(CycleGAN_G, self).__init__()
        ### Downsample block
        ## Reflection padding is an alternative to 0 padding (like looking at water reflection)
        # n_colors x image_size x image_size
        model = [nn.ReflectionPad2d(padding=3),
                nn.Conv2d(param.n_colors, param.G_h_size, kernel_size=7, stride=1, padding=0),
                Norm2D(param.G_h_size),
                nn.ReLU(True)]
        # param.G_h_size x image_size x image_size
        model += [nn.Conv2d(param.G_h_size, param.G_h_size * 2, kernel_size=3, stride=2, padding=1),
                Norm2D(param.G_h_size * 2),
                nn.ReLU(True)]
        # (param.G_h_size * 2) x (image_size / 2) x (image_size / 2)
        model += [nn.Conv2d(param.G_h_size * 2, param.G_h_size * 4, kernel_size=3, stride=2, padding=1),
                Norm2D(param.G_h_size * 4),
                nn.ReLU(True)]
        # (param.G_h_size * 4) x (image_size / 4) x (image_size / 4)

        ### Residual blocks
        for i in range(param.G_residual_blocks):
            model += [Residual_block(h_size=param.G_h_size * 4)]

        ### Upsample block (pretty much inverse of downsample)
        # (param.G_h_size * 4) x (image_size / 4) x (image_size / 4)
        model += [nn.ConvTranspose2d(param.G_h_size * 4, param.G_h_size * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                Norm2D(param.G_h_size * 2),
                nn.ReLU(True)]
        # (param.G_h_size * 2) x (image_size / 2) x (image_size / 2)
        model += [nn.ConvTranspose2d(param.G_h_size * 2, param.G_h_size, kernel_size=3, stride=2, padding=1, output_padding=1),
                Norm2D(param.G_h_size),
                nn.ReLU(True)]
        # param.G_h_size x image_size x image_size
        model += [nn.ReflectionPad2d(padding=3),
                nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=7, stride=1, padding=0),
                nn.Tanh()]
        # Size = n_colors x image_size x image_size
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(param.n_gpu))
        else:
            output = self.model(input)
        return output

# Discriminator (70 x 70 PatchGAN)
# A PatchGAN try to classify N x N patches of an image instead of an image itself. We output the average
# Although not specified in the paper; in their official code, they use 2 zero-padding and also stride 1 in two last layers
# Using 1 zero-padding instead of 2 here will get you a simple 1 x 32 x 32 output when image_size = 256 x 256, otherwise it's 1 x 27 x 27
class CycleGAN_D(torch.nn.Module):
    def __init__(self):
        super(CycleGAN_D, self).__init__()

        # Size = n_colors x image_size x image_size
        model = [nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True)]
        # Size = D_h_size x (image_size / 2) x (image_size / 2)
        model += [nn.Conv2d(param.D_h_size, param.D_h_size * 2, kernel_size=4, stride=2, padding=2),
                Norm2D(param.D_h_size * 2),
                nn.LeakyReLU(0.2, inplace=True)]
        # Size = (D_h_size * 2) x (image_size / 4) x (image_size / 4)
        model += [nn.Conv2d(param.D_h_size * 2, param.D_h_size * 4, kernel_size=4, stride=2, padding=2),
                Norm2D(param.D_h_size * 4),
                nn.LeakyReLU(0.2, inplace=True)]
        # Size = (D_h_size * 4) x (image_size / 8) x (image_size / 8)
        model += [nn.Conv2d(param.D_h_size * 4, param.D_h_size * 8, kernel_size=4, stride=1, padding=2),
                Norm2D(param.D_h_size * 8),
                nn.LeakyReLU(0.2, inplace=True)]
        # Size = (D_h_size * 8) x (image_size / 8) x (image_size / 8)
        model += [nn.Conv2d(param.D_h_size * 8, 1, kernel_size=2, stride=1, padding=2)]
        # Size = 1 x (image_size / 8)) x (image_size / 8)
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(param.n_gpu))
        else:
            output = self.model(input)
        return output

## Weights init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

## Initialization
G1 = CycleGAN_G()
G2 = CycleGAN_G()
if not param.trained_model:
    D1 = CycleGAN_D()
    D2 = CycleGAN_D()

# Initialize weights
G1.apply(weights_init)
G2.apply(weights_init)
if not param.trained_model:
    D1.apply(weights_init)
    D2.apply(weights_init)

# Load existing models
if param.Path_load != '':
    G1.load_state_dict(torch.load(param.Path_load.replace("X_epoch_","G1_epoch_")))
    G2.load_state_dict(torch.load(param.Path_load.replace("X_epoch_","G2_epoch_")))
    if not param.trained_model:
        D1.load_state_dict(torch.load(param.Path_load.replace("X_epoch_","D1_epoch_")))
        D2.load_state_dict(torch.load(param.Path_load.replace("X_epoch_","D2_epoch_")))

print(G1)
print(G1, file=log_output)
if not param.trained_model:
    print(D1)
    print(D1, file=log_output)

# Criterions (L1Loss for cycle consistency losses, MSELoss for other losses)
# Note: Could have written those manually like I did before, but I wanted to try using the torch loss functions
criterion = nn.MSELoss()
criterion_cycle = nn.L1Loss()

## Soon to be variables
x1 = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
x2 = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)

# Everything cuda
if param.cuda:
    G1 = G1.cuda()
    G2 = G2.cuda()
    if not param.trained_model:
        D1 = D1.cuda()
        D2 = D2.cuda()
    x1 = x1.cuda()
    x2 = x2.cuda()
    criterion = criterion.cuda()
    criterion_cycle = criterion_cycle.cuda()

# Now Variables
x1 = Variable(x1)
x2 = Variable(x2)

#### If trained, we only generate all pictures
if param.trained_model:
    for i, ((image1, label1), (image2, label1)) in enumerate(dataset):
        # To prevent out-of-memory errors, we only do one of each domain at a time
        for domain in range(1,2):
            # Getting current image
            if domain == 1: image = image1
            if domain == 2: image = image2
            if param.cuda:
                image = image.cuda()
            x = Variable(image)
            # Generate and save fake image
            if domain == 1: x_fake = G1(x)
            if domain == 2: x_fake = G2(x)
            vutils.save_image(x_fake.data, '%s/run-%d/images/fake_samples%d_%05d.png' % (param.output_folder, run, domain, i), normalize=True)
    quit()

# Adam optimizer
optimizerD1 = torch.optim.Adam(D1.parameters(), lr=param.lr, betas=(param.beta1, param.beta2))
optimizerD2 = torch.optim.Adam(D2.parameters(), lr=param.lr, betas=(param.beta1, param.beta2))
# We chain both G1 and G2 together so optimizer will train them simultaneously
# Why? : Because we need to train G1 and G2 with cycle consistency |G1(G2(x2)) - x2|  and |G2(G1(x1)) - x1|
# We could either do it with alternating optimization or do them together, it's best to simply do them together.
optimizerG = torch.optim.Adam(itertools.chain(G1.parameters(), G2.parameters()), lr=param.lr, betas=(param.beta1, param.beta2))

## Fake image Pool
# In CycleGAN instead of simply always feeding the current fakes, they replace the fake images with a previous fake image (from the image pool) with prob = 1/2.
# This approach comes from https://arxiv.org/pdf/1612.07828.pdf, it improves the stability of GANs with image_A -> image_B (As opposed to the usual random_numbers -> fake_image)
# This is a copy of their own class at : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py
class FakeImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

# Initialize fake image pools
Fake1_pool = FakeImagePool(param.pool_size)
Fake2_pool = FakeImagePool(param.pool_size)

## Fitting model
for epoch in range(param.n_epoch+param.n_epoch_decay):

    # Learning rate decay as per paper
    if epoch >= param.n_epoch:
        lr = param.lr * ((param.n_epoch + param.n_epoch_decay - epoch) / param.n_epoch_decay)
        for param_group in optimizerD1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizerD2.param_groups:
            param_group['lr'] = lr
        for param_group in optimizerG.param_groups:
            param_group['lr'] = lr

    for i, ((images1, label1), (images2, label1)) in enumerate(dataset):

        ### Getting current images
        if param.cuda:
            images1 = images1.cuda()
            images2 = images2.cuda()
        x1.data.copy_(images1)
        x2.data.copy_(images2)

        #########################
        # (1) Update G networks #
        #########################

        G1.zero_grad()
        G2.zero_grad()

        # Pretending that these are not a fake to bamboozle the Discriminators
        x1_fake = G2(x2)
        y1_pred_fake = D1(x1_fake)

        # Labels
        # I need to initialize them here because any slight change in the Discriminator structure will change the output size
        # This is because of PatchGAN trying to predict real or fake in each patchs of the image
        # The size of D1 and D2 output is batch_size x 1 x patch_Size x patch_Size (where patch_size depends on the hyperparameters of D1 and D2)
        if i == 0 and epoch == 0:
            one = torch.FloatTensor(y1_pred_fake.size()).fill_(1)
            zero = torch.FloatTensor(y1_pred_fake.size()).fill_(0)
            if param.cuda:
                one = one.cuda()
                zero = zero.cuda()
            one = Variable(one, requires_grad=False)
            zero = Variable(zero, requires_grad=False)


        errG1 = criterion(y1_pred_fake, one)
        x2_fake = G1(x1)
        y2_pred_fake = D2(x2_fake)
        errG2 = criterion(y1_pred_fake, one)

        # Cycle consistency losses
        x1_fake_cycle = G2(G1(x1))
        errCyc1 = param.penalty*criterion_cycle(x1_fake_cycle, x1)
        x2_fake_cycle = G1(G2(x2))
        errCyc2 = param.penalty*criterion_cycle(x2_fake_cycle, x2)

        # Identity losses
        errId1 = param.identity_penalty*criterion_cycle(G2(x1), x1)
        errId2 = param.identity_penalty*criterion_cycle(G1(x2), x2)

        # Backward step and optimizing
        errG_total = errG1 + errG2 + errCyc1 + errCyc2 + errId1 + errId2
        errG_total.backward()
        optimizerG.step()

        #########################
        # (2) Update D1 network #
        #########################

        D1.zero_grad()

        # Real images
        y1_pred = D1(x1)
        errD1_real = criterion(y1_pred, one)

        # Fake images
        x1_fake_ = Fake1_pool.query(x1_fake)
        y1_pred_fake = D1(x1_fake_)
        errD1_fake = criterion(y1_pred_fake, zero)

        # Backward step and optimizing
        errD1_total = 0.5 * (errD1_real + errD1_fake)
        errD1_total.backward()
        optimizerD1.step()

        #########################
        # (3) Update D2 network #
        #########################

        D2.zero_grad()

        # Real images
        y2_pred = D2(x2)
        errD2_real = criterion(y2_pred, one)

        # Fake images
        x2_fake_ = Fake2_pool.query(x2_fake)
        y2_pred_fake = D2(x2_fake_)
        errD2_fake = criterion(y2_pred_fake, zero)

        # Backward step and optimizing
        errD2_total = 0.5 * (errD2_real + errD2_fake)
        errD2_total.backward()
        optimizerD2.step()

        ################################################
        # (4) Logging, printing and saving fake images #
        ################################################

        errD_total = errD1_total + errD2_total
        current_step = i + epoch*len(dataset)
        # Log results so we can see them in TensorBoard after

        log_value('errD', errD_total.data[0], current_step)
        log_value('errG', errG_total.data[0], current_step)

        if current_step % 50 == 0:
            end = time.time()
            fmt = '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f time:%.4f'
            s = fmt % (epoch, param.n_epoch + param.n_epoch_decay, i, len(dataset), errD_total.data[0], errG_total.data[0], end - start)
            print(s)
            print(s, file=log_output)

        if current_step % 50 == 0:
            #vutils.save_image(x1.data, '%s/run-%d/images/true_samples1_epoch%03d_iter%03d.png' % (param.output_folder, run, epoch, i), normalize=True)
            #vutils.save_image(x2.data, '%s/run-%d/images/true_samples2_epoch%03d_iter%03d.png' % (param.output_folder, run, epoch, i), normalize=True)
            vutils.save_image(x1_fake.data, '%s/run-%d/images/fake_samples1_epoch%03d_iter%03d.png' % (param.output_folder, run, epoch, i), normalize=True)
            vutils.save_image(x2_fake.data, '%s/run-%d/images/fake_samples2_epoch%03d_iter%03d.png' % (param.output_folder, run, epoch, i), normalize=True)
    # Save every epoch
    fmt = '%s/run-%d/models/%s_epoch_%d.pth'
    if epoch % 10 == 0:
        torch.save(G1.state_dict(), fmt % (param.output_folder, run, 'G1', epoch))
        torch.save(G2.state_dict(), fmt % (param.output_folder, run, 'G2', epoch))
        torch.save(D1.state_dict(), fmt % (param.output_folder, run, 'D1', epoch))
        torch.save(D2.state_dict(), fmt % (param.output_folder, run, 'D2', epoch))
