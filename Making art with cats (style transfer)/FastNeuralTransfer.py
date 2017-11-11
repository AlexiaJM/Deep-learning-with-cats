# Reference 1 : https://arxiv.org/pdf/1603.08155.pdf
# Reference 2 : https://www.slideshare.net/misgod/fast-neural-style
# Reference 3 : https://github.com/jcjohnson/fast-neural-style
# Reference 4 : https://github.com/abhiskk/fast-neural-style/

# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/DCGAN

## You need COCO Dataset extracted in the input_folder, to get it, run this:
# wget -nc http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# unzip train2014.zip

## I highly recommend using "--NN_conv True"

#### Example use
## Training
# python FastNeuralTransfer.py --seed 77 --H_size 16 --batch_size 8 --style_image_size 512 --ANTIALIAS True --style_weight 5 --NN_conv True --style_picture style/picasso.jpg
## Applying filter to cats
# python FastNeuralTransfer.py --NN_conv True --model_load /FNT/models/model_end.pth --trained_model True --input_folder /Datasets/Meow --seed 3235

#### More examples from my own use
## Rain Princess 256, NN_conv
# python FastNeuralTransfer.py --seed 77 --H_size 16 --batch_size 8 --style_image_size 256 --ANTIALIAS True --style_weight 10 --NN_conv True --style_picture /mnt/sdb2/styles/rain-princess.jpg
# python FastNeuralTransfer.py --NN_conv True --model_load /mnt/sdb2/Output/FNT/rainyday_256size_stylew4_ANTIALI_NNConv/models/model_end.pth --trained_model True --input_folder /home/alexia/Datasets/Meow
# python FastNeuralTransfer.py --NN_conv True --model_load /mnt/sdb2/Output/FNT/rainyday_256size_stylew4_ANTIALI_NNConv/models/model_end.pth --trained_model True --input_folder /mnt/sdb2/bibu_pictures
## Starry Night
#python FastNeuralTransfer.py --seed 77 --H_size 16 --batch_size 8 --style_image_size 512 --ANTIALIAS True --style_weight 10 --NN_conv True --style_picture /mnt/sdb2/styles/starry_night-crop2.jpg
#python FastNeuralTransfer.py --NN_conv True --model_load /mnt/sdb2/Output/FNT/starry_512size_stylew10_ANTIALI_NNConv/models/model_end.pth --trained_model True --input_folder /home/alexia/Datasets/Meow
#python FastNeuralTransfer.py --NN_conv True --model_load /mnt/sdb2/Output/FNT/starry_512size_stylew10_ANTIALI_NNConv/models/model_end.pth --trained_model True --input_folder /mnt/sdb2/bibu_pictures


## Parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=256, help='Image size of training images')
parser.add_argument('--style_image_size', type=int, default=256, help='Image size of the style (if equal to 0, will not resize). Resizing leads to very different results, I recommended using 256, 512 or 0 and see which one you prefer.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size, paper use 4.')
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--H_size', type=int, default=16, help='Number of filters in the image transformator. Paper use 32 but official implementation recommend only using 16 for faster speed while retaining quality.')
parser.add_argument('--Residual_blocks', type=int, default=5, help='Number of residual blocks')
parser.add_argument('--SELU', type=bool, default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. Do not use.')
parser.add_argument('--norm_type', default='instance', help='If "instance" uses instance normalization; if "batch" uses batch normalization.')
parser.add_argument('--padding', default='reflect', help='If "reflect" uses reflection padding; if "zero" uses zero padding.')
parser.add_argument('--lr', type=float, default=.001, help='Learning rate')
parser.add_argument('--n_epoch', type=int, default=2, help='Number of epochs. 2 is generally enough, but more can be better, just Ctrl-C to stop it when you feel like it is good enough.')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam betas[0]')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
parser.add_argument("--content_weight", type=float, default=1, help="Weight of content loss")
parser.add_argument("--style_weight", type=float, default=5, help="Weight of style loss. Make bigger or smaller depending on observed results vs desired results.")
parser.add_argument("--total_variation_weight", type=float, default=1e-6, help="Weight of total variation loss (Should be between 1e-4 and 1e-6)")
parser.add_argument("--feature", type=int, default=1, help="Contant loss feature used: 0=relu1_2, 1=relu2_2, 2=relu3_3, 3=relu4_3. Paper use relu2_2, official implementation use relu3_3.")
parser.add_argument("--NN_conv", type=bool, default=False, help="This is highly recommended. This approach minimize checkerboard artifacts during training. Uses nearest-neighbor resized convolutions instead of strided convolutions (https://distill.pub/2016/deconv-checkerboard/ and github.com/abhiskk/fast-neural-style).")
parser.add_argument("--ANTIALIAS", type=bool, default=False, help="Use antialiasing instead of bilinear to resize images. Sightly slower but sightly better quality.")
parser.add_argument('--seed', type=int)
parser.add_argument('--input_folder', default='/mnt/sdb2/Datasets/COCO', help='input folder (Coco dataset for training, whichever dataset after to apply style)')
parser.add_argument('--style_picture', default='/mnt/sdb2/styles/candy.jpg', help='Style picture')
parser.add_argument('--VGG16_folder', default='/mnt/sdb2/VGG16', help='folder for the VGG16 (Will be downloaded automatically into this folder)')
parser.add_argument('--output_folder', default='/mnt/sdb2/Output/FNT', help='output folder')
parser.add_argument('--model_load', default='', help='Full path to transformation model to load (ex: /home/output_folder/run-5/models/G_epoch_11.pth)')
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
import torch.nn.functional as F

# For plotting the Loss of D and G using tensorboard
from tensorboard_logger import configure, log_value
configure(logs_dir, flush_secs=5)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import PIL

from torch.utils.serialization import load_lua

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
trans = []
if not param.trained_model:
    if param.ANTIALIAS:
        trans.append(transf.Scale(param.image_size,PIL.Image.ANTIALIAS))
    else:
        trans.append(transf.Scale(param.image_size,PIL.Image.BILINEAR))
    trans.append(transf.CenterCrop(param.image_size))
trans.append(transf.ToTensor())
trans.append(transf.Lambda(lambda x: x.mul(255)))
trans = transf.Compose(trans)

trans_style = []
if param.style_image_size > 0:
    if param.ANTIALIAS:
        trans_style.append(transf.Scale(param.style_image_size,PIL.Image.ANTIALIAS))
    else:
        trans_style.append(transf.Scale(param.style_image_size,PIL.Image.BILINEAR))
    trans_style.append(transf.CenterCrop(param.style_image_size))
trans_style.append(transf.ToTensor())
trans_style.append(transf.Lambda(lambda x: x.mul(255)))
trans_style = transf.Compose(trans_style)

## Importing datasets
data = dset.ImageFolder(root=param.input_folder, transform=trans)

# Loading data in batch
if param.trained_model:
    param.batch_size = 1
dataset = torch.utils.data.DataLoader(data, batch_size=param.batch_size, shuffle=True, num_workers=param.n_workers, drop_last=True)

# Style image
style_picture = PIL.Image.open(param.style_picture)
style_picture = trans_style(style_picture)
vutils.save_image(style_picture.repeat(1, 1, 1, 1), '%s/run-%d/images/style.png' % (param.output_folder, run), normalize=True)

## Models
# Reflection padding is an alternative to 0 padding (like looking at water reflection)
# Official implementation of the paper use only reflection paddding in the downsample block but I use it everywhere, I'm not sure if it makes a difference.

# Set batch norm or instance norm
import functools
if param.norm_type == 'batch':
    Norm2D = functools.partial(nn.BatchNorm2d)
elif param.norm_type == 'instance':
    Norm2D = functools.partial(nn.InstanceNorm2d)

# Padding
if param.padding == "reflect":
    pad = 0
if param.padding == "zero":
    pad = 1

# Residual Block of Generator
class Residual_block(torch.nn.Module):
    def __init__(self, h_size):
        super(Residual_block, self).__init__()
        # Two Conv layers with same output size
        model = []
        if param.padding == "reflect":
            model += [nn.ReflectionPad2d(padding=1)]
        model += [nn.Conv2d(h_size, h_size, kernel_size=3, stride=1, padding=pad)]
        if param.SELU:
            model += [torch.nn.SELU()]
        else:
            model += [Norm2D(h_size),
                    nn.ReLU(True)]
        if param.padding == "reflect":
            model += [nn.ReflectionPad2d(padding=1)]
        model += [nn.Conv2d(h_size, h_size, kernel_size=3, stride=1, padding=pad)]
        if not param.SELU:
            model += [Norm2D(h_size)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # Return itself + the result of the two convolutions
        output = self.model(input) + input
        return output

# Image transformation network
class Image_transform_net(torch.nn.Module):
    def __init__(self):
        super(Image_transform_net, self).__init__()
        model = []
        ### Downsample block
        # n_colors x image_size x image_size
        if param.padding == "reflect":
            model += [nn.ReflectionPad2d(padding=4)]
        model += [nn.Conv2d(param.n_colors, param.H_size, kernel_size=9, stride=1, padding=pad*4)]
        if param.SELU:
            model += [torch.nn.SELU()]
        else:
            model += [Norm2D(param.H_size),
                    nn.ReLU(True)]
        # param.H_size x image_size x image_size
        if param.padding == "reflect":
            model += [nn.ReflectionPad2d(padding=1)]
        model += [nn.Conv2d(param.H_size, param.H_size * 2, kernel_size=3, stride=2, padding=pad)]
        if param.SELU:
            model += [torch.nn.SELU()]
        else:
            model += [Norm2D(param.H_size * 2),
                    nn.ReLU(True)]
        # (param.H_size * 2) x (image_size / 2) x (image_size / 2)
        if param.padding == "reflect":
            model += [nn.ReflectionPad2d(padding=1)]
        model += [nn.Conv2d(param.H_size * 2, param.H_size * 4, kernel_size=3, stride=2, padding=pad)]
        if param.SELU:
            model += [torch.nn.SELU()]
        else:
            model += [Norm2D(param.H_size * 4),
                    nn.ReLU(True)]
        # (param.H_size * 4) x (image_size / 4) x (image_size / 4)

        ### Residual blocks
        for i in range(param.Residual_blocks):
            model += [Residual_block(h_size=param.H_size * 4)]

        ### Upsample block
        # (param.H_size * 4) x (image_size / 4) x (image_size / 4)
        if param.NN_conv:
            model += [nn.Upsample(scale_factor=2)]
            if param.padding == "reflect":
                model += [nn.ReflectionPad2d(padding=1)]
            model += [torch.nn.Conv2d(param.H_size * 4, param.H_size * 2, kernel_size=3, stride=1, padding=pad)]
        else:
            if param.padding == "reflect":
                model += [nn.ReflectionPad2d(padding=1)]
            model += [nn.ConvTranspose2d(param.H_size * 4, param.H_size * 2, kernel_size=3, stride=2, padding=pad, output_padding=1)]
        if param.SELU:
            model += [torch.nn.SELU()]
        else:
            model += [Norm2D(param.H_size * 2),
                    nn.ReLU(True)]
        # (param.H_size * 2) x (image_size / 2) x (image_size / 2)
        if param.NN_conv:
            model += [nn.Upsample(scale_factor=2)]
            if param.padding == "reflect":
                model += [nn.ReflectionPad2d(padding=1)]
            model += [torch.nn.Conv2d(param.H_size * 2, param.H_size, kernel_size=3, stride=1, padding=pad)]
        else:
            model += [nn.ConvTranspose2d(param.H_size * 2, param.H_size, kernel_size=3, stride=2, padding=1, output_padding=1)]
        if param.SELU:
            model += [torch.nn.SELU()]
        else:
            model += [Norm2D(param.H_size),
                    nn.ReLU(True)]
        # param.H_size x image_size x image_size
        if param.padding == "reflect":
            model += [nn.ReflectionPad2d(padding=4)]
        model += [nn.Conv2d(param.H_size, param.n_colors, kernel_size=9, stride=1, padding=pad*4),
                nn.Tanh()]
        # Size = n_colors x image_size x image_size
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(param.n_gpu))
        else:
            output = self.model(input)
        # Contradictions between paper and official implementation... one says output scaled to [0,255] and the other scale by 150
        # Some also don't even use tanh... (https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py)
        # Going to scale [0,255] so that we can simply substract means of imagenet
        output = ((output+1)/2)*255
        return output

# VGG16
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # We need to save those and do the forward step manually to be able to keep every in-between steps
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Remove gradients since we won't need them
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        return [relu1_2, relu2_2, relu3_3, relu4_3]

## Weights init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

## Initialization
TNet = Image_transform_net()
if not param.trained_model:
    VGG16 = VGG16()
    # VGG-16 working on [0,255] scale
    # Can't use PyTorch trained model (https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py) because we need access to in-between steps
    # Need to use Justin Johnson Lua trained model and convert it
    os.chdir(param.VGG16_folder)
    os.system('wget -nc http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7')
    VGG16_Lua = load_lua('vgg16.t7')
    for (src, dst) in zip(VGG16_Lua.parameters()[0], VGG16.parameters()):
        # dst[:].data = src[:]
        dst.data[:] = src
    torch.save(VGG16.state_dict(), f"{base_dir}/VGG16.pth")
# Initialize weights
TNet.apply(weights_init)

# Load existing models
if param.model_load != '':
    TNet.load_state_dict(torch.load(param.model_load))

print(TNet)
print(TNet, file=log_output)
if not param.trained_model:
    print(VGG16)
    print(VGG16, file=log_output)

# Criterion
criterion = nn.MSELoss()

###### Functions for processing

## Gram Matrix
# This is what is used to assess style of the picture, when normalized, it represent correlations between every features
# corr(X,X) = Var(X)/sigma(X)^2 = E[(X-u_x)*t(X-u_x)]/sigma(X)^2 = E[(Y)*t(Y)] where Y = (X-u_x)/sigma(X)
def gram_matrix(y):
    # (batch_size, n_colors, height, width)
    (batch_size, n_colors, height, width) = y.size()
    features = y.view(batch_size, n_colors, height * width)
    # Transpose matrix for the two last columns (ignoring batch_size)
    features_t = features.transpose(1, 2)
    # Y * t(Y) and scaling (Very important to scale because style and images are generally not the same size and param.style_weight is set assuming loss is scaled)
    gram = features.bmm(features_t) / (n_colors * height * width)
    return gram

## Substract mean from image net to be able to enter in VGG16
def substract_mean(batch):
    mean = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch -= Variable(mean)
    return batch

## VGG16 was not only trained mean centered but also using BGR instead of RBG so we need change all pictures
# Note this works both way, RGB -> BGR and BGR -> RGB
def RGB_to_BGR(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

## Soon to be variables
x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
style = style_picture.repeat(param.batch_size, 1, 1, 1)

# Everything cuda
if param.cuda:
    TNet = TNet.cuda()
    if not param.trained_model:
        VGG16 = VGG16.cuda()
    x = x.cuda()
    style = style.cuda()
    criterion = criterion.cuda()

# Now Variables
x = Variable(x)
style = RGB_to_BGR(style)
style = Variable(style, volatile=True)

#### If trained, we only generate all pictures
if param.trained_model:
    for i, (image, label) in enumerate(dataset):
        # Getting current image
        if param.cuda:
            image = image.cuda()
        image = Variable(image, volatile=True)
        # Generate and save stylized image
        x_transf = TNet(image)
        x_transf = RGB_to_BGR(x_transf)
        vutils.save_image(x_transf.data, '%s/run-%d/images/stylized_samples_%05d.png' % (param.output_folder, run, i), normalize=True, padding=0)
    quit()

## Getting style image Gram Matrix
style = substract_mean(style)
features_style = VGG16(style)
gram_style = [gram_matrix(y) for y in features_style]
del style
del features_style

# Adam optimizer
optimizerTNet = torch.optim.Adam(TNet.parameters(), lr=param.lr, betas=(param.beta1, param.beta2))

## Fitting model
for epoch in range(param.n_epoch):
    for i, (images, label) in enumerate(dataset):

        optimizerTNet.zero_grad()

        ### Getting current images
        if param.cuda:
            images = images.cuda()
        images = RGB_to_BGR(images)
        x.data.copy_(images)
        # Transformation
        x_transf = TNet(x)
        # Getting features
        x = substract_mean(x)
        features_x = VGG16(x)
        x_transf = substract_mean(x_transf)
        features_x_transf = VGG16(x_transf)

        ## Loss of Content
        ## We only use the second feature (relu2_2) as they did in the paper
        Loss_content = param.content_weight*criterion(features_x_transf[param.feature], features_x[param.feature])

        ## Loss of Style
        Loss_style = 0
        for g_s, f_x_transf in zip(gram_style, features_x_transf):
            gram_x_transf = gram_matrix(f_x_transf)
            #(batch_size, n_colors, height, width) = f_x_transf.size()
            Loss_style += criterion(gram_x_transf, Variable(g_s.data, requires_grad=False))
        Loss_style = param.style_weight*Loss_style

        ## Loss of total variation
        # Not clear what is the exponent of this loss function (https://arxiv.org/pdf/1412.0035.pdf use 1/2 and 3/2)
        # I'll assume it's 1.
        # https://arxiv.org/pdf/1412.0035.pdf says to sum for each color
        x_transf_size = x_transf.size()
        x_i_diff = (x_transf[:, :, :(x_transf_size[2] - 1), :(x_transf_size[3] - 1)] - x[:, :, :(x_transf_size[2] - 1), 1:]) ** 2
        x_j_diff = (x_transf[:, :, :(x_transf_size[2] - 1), :(x_transf_size[3] - 1)] - x[:, :, 1:, :(x_transf_size[3] - 1)]) ** 2
        # Sum over n_colors, weidth, height and average over batch_size
        Loss_tv = param.total_variation_weight*((x_i_diff + x_j_diff).sum(3).sum(2).sum(1).mean())

        ## Total Loss
        Loss = Loss_content + Loss_style + Loss_tv

        Loss.backward()
        optimizerTNet.step()

        current_step = i + epoch*len(dataset)
        # Log results so we can see them in TensorBoard after
        log_value('errContent', Loss_content.data[0], current_step)
        log_value('errStyle', Loss_style.data[0], current_step)
        log_value('errTV', Loss_tv.data[0], current_step)
        log_value('errTotal', Loss.data[0], current_step)

        if current_step % 100 == 0:
            end = time.time()
            fmt = '[%d/%d][%d/%d] Loss_Total %.4f Loss_Content: %.4f Loss_Style: %.4f Loss_TV: %.4f time:%.4f'
            s = fmt % (epoch, param.n_epoch, i, len(dataset), Loss.data[0], Loss_content.data[0], Loss_style.data[0], Loss_tv.data[0], end - start)
            print(s)
            print(s, file=log_output)
            #x = RGB_to_BGR(x)
            #vutils.save_image(x.data, '%s/run-%d/images/real_samples_%03d.png' % (param.output_folder, run, current_step/50), normalize=True)
            x_transf = RGB_to_BGR(x_transf)
            vutils.save_image(x_transf.data, '%s/run-%d/images/stylized_samples_%03d.png' % (param.output_folder, run, current_step/50), normalize=True)
        if current_step % 1000 == 0:
            fmt = '%s/run-%d/models/model_epoch_%d_iter_%03d.pth'
            torch.save(TNet.state_dict(), fmt % (param.output_folder, run, epoch, i))
    # Save at the end
    fmt = '%s/run-%d/models/model_end.pth'
    torch.save(TNet.state_dict(), fmt % (param.output_folder, run))
