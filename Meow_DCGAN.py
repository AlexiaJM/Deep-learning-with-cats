# Reference 1 : https://github.com/pytorch/examples/blob/master/dcgan
# Reference 2 : https://arxiv.org/pdf/1511.06434.pdf
# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/DCGAN

## Parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64) # DCGAN paper original value used 128
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=100) # DCGAN paper original value
parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in the Generator. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--D_h_size', type=int, default=128, help='Number of hidden nodes in the Discriminator. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--lr_D', type=float, default=.00005, help='Discriminator learning rate') # 1/4 of DCGAN paper original value
parser.add_argument('--lr_G', type=float, default=.0002, help='Generator learning rate') # DCGAN paper original value
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
parser.add_argument('--SELU', type=bool, default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. This improves stability.')
parser.add_argument('--seed', type=int)
parser.add_argument('--input_folder', default='/home/alexia/Datasets/Meow_64x64', help='input folder')
parser.add_argument('--output_folder', default='/home/alexia/Output/DCGAN', help='output folder')
parser.add_argument('--G_load', default='', help='Full path to Generator model to load (ex: /home/output_folder/run-5/models/G_epoch_11.pth)')
parser.add_argument('--D_load', default='', help='Full path to Discriminator model to load (ex: /home/output_folder/run-5/models/D_epoch_11.pth)')
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
while os.path.exists("%s/run-%d" % (param.output_folder, run)):
	run += 1
os.mkdir("%s/run-%d" % (param.output_folder, run))
os.mkdir("%s/run-%d/logs" % (param.output_folder, run))
os.mkdir("%s/run-%d/images" % (param.output_folder, run))
os.mkdir("%s/run-%d/models" % (param.output_folder, run))

# where we save the output
log_output = open("%s/run-%d/logs/log.txt" % (param.output_folder, run), 'w')
print(param)
print(param, file=log_output)

import torch
import torch.autograd as autograd
from torch.autograd import Variable

# For plotting the Loss of D and G using tensorboard
from tensorboard_logger import configure, log_value
configure("%s/run-%d/logs" % (param.output_folder, run), flush_secs=5)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils

if param.cuda:
	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True

# To see images
from IPython.display import Image
to_img = transf.ToPILImage()

import math

## Setting seed
import random
if param.seed is None:
	param.seed = random.randint(1, 10000)
print("Random Seed: ", param.seed)
print("Random Seed: ", param.seed, file=log_output)
random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
	torch.cuda.manual_seed_all(param.seed)

## Transforming images
trans = transf.Compose([
	transf.Scale((param.image_size, param.image_size)),
	# This makes it into [0,1]
	transf.ToTensor(),
	# This makes it into [-1,1] so tanh will work properly
	transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

## Importing dataset
data = dset.ImageFolder(root=param.input_folder, transform=trans)

# Loading data in batch
dataset = torch.utils.data.DataLoader(data, batch_size=param.batch_size, shuffle=True, num_workers=param.n_workers)

## Models
# The number of layers is implicitly determined by the image size
# image_size = (4,8,16,32,64, 128, 256, 512, 1024) leads to n_layers = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# The more layers the bigger the neural get so it's best to decrease G_h_size and D_h_size when the image input is bigger

# DCGAN generator
class DCGAN_G(torch.nn.Module):
	def __init__(self):
		super(DCGAN_G, self).__init__()
		main = torch.nn.Sequential()

		# We need to know how many layers we will use at the beginning
		mult = param.image_size // 8

		### Start block
		# Z_size random numbers
		main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
		if param.SELU:
			main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
		else:
			main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
			main.add_module('Start-ReLU', torch.nn.ReLU())
		# Size = (G_h_size * mult) x 4 x 4

		### Middle block (Done until we reach ? x image_size/2 x image_size/2)
		i = 1
		while mult > 1:
			main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
			if param.SELU:
				main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
			else:
				main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
				main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
			# Size = (G_h_size * (mult/(2*i))) x 8 x 8
			mult = mult // 2
			i += 1

		### End block
		# Size = G_h_size x image_size/2 x image_size/2
		main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False))
		main.add_module('End-Tanh', torch.nn.Tanh())
		# Size = n_colors x image_size x image_size
		self.main = main

	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
		else:
			output = self.main(input)
		return output

# DCGAN discriminator (using somewhat the reverse of the generator)
class DCGAN_D(torch.nn.Module):
	def __init__(self):
		super(DCGAN_D, self).__init__()
		main = torch.nn.Sequential()

		### Start block
		# Size = n_colors x image_size x image_size
		main.add_module('Start-Conv2d', torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
		if param.SELU:
			main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
		else:
			main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
		image_size_new = param.image_size // 2
		# Size = D_h_size x image_size/2 x image_size/2

		### Middle block (Done until we reach ? x 4 x 4)
		mult = 1
		i = 0
		while image_size_new > 4:
			main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
			if param.SELU:
				main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
			else:
				main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.D_h_size * (2*mult)))
				main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
			# Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
			image_size_new = image_size_new // 2
			mult = mult*2
			i += 1

		### End block
		# Size = (D_h_size * mult) x 4 x 4
		main.add_module('End-Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
		main.add_module('End-Sigmoid', torch.nn.Sigmoid())
		# Size = 1 x 1 x 1 (Is a real cat or not?)
		self.main = main

	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
		else:
			output = self.main(input)
		# Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
		return output.view(-1)

## Weights init function, DCGAN use 0.02 std
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
G = DCGAN_G()
D = DCGAN_D()

# Initialize weights
G.apply(weights_init)
D.apply(weights_init)

# Load existing models
if param.G_load != '':
	G.load_state_dict(torch.load(param.G_load))
if param.D_load != '':
	D.load_state_dict(torch.load(param.D_load))

print(G)
print(G, file=log_output)
print(D)
print(D, file=log_output)

# Criterion
criterion = torch.nn.BCELoss()

# Soon to be variables
x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
y = torch.FloatTensor(param.batch_size)
z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
# This is to see during training, size and values won't change
z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1)

# Everything cuda
if param.cuda:
	G = G.cuda()
	D = D.cuda()
	criterion = criterion.cuda()
	x = x.cuda()
	y = y.cuda()
	z = z.cuda()
	z_test = z_test.cuda()

# Now Variables
x = Variable(x)
y = Variable(y)
z = Variable(z)
z_test = Variable(z_test)

# Based on DCGAN paper, they found using betas[0]=.50 better.
# betas[0] represent is the weight given to the previous mean of the gradient
# betas[1] is the weight given to the previous variance of the gradient
optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, 0.999))

## Fitting model
for epoch in range(param.n_epoch):
	for i, data_batch in enumerate(dataset, 0):
		########################
		# (1) Update D network #
		########################

		for p in D.parameters():
			p.requires_grad = True

		# Train with real data
		D.zero_grad()
		# We can ignore labels since they are all cats!
		images, labels = data_batch
		# Mostly necessary for the last one because if N might not be a multiple of batch_size
		current_batch_size = images.size(0)
		if param.cuda:
			images = images.cuda()
		# Transfer batch of images to x
		x.data.resize_as_(images).copy_(images)
		# y is now a vector of size current_batch_size filled with 1
		y.data.resize_(current_batch_size).fill_(1)
		y_pred = D(x)
		errD_real = criterion(y_pred, y)
		errD_real.backward()
		# Var has data and gradient element, we keep the mean of the data element
		D_real = y_pred.data.mean()

		# Train with fake data
		z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
		x_fake = G(z)
		y.data.resize_(current_batch_size).fill_(0)
		# Detach y_pred from the neural network G and put it inside D
		y_pred_fake = D(x_fake.detach())
		errD_fake = criterion(y_pred_fake, y)
		errD_fake.backward()
		D_fake = y_pred_fake.data.mean()
		errD = errD_real + errD_fake
		optimizerD.step()

		########################
		# (2) Update G network #
		########################

		# Make it a tiny bit faster
		for p in D.parameters():
			p.requires_grad = False

		G.zero_grad()
		# Generator wants to fool discriminator so it wants to minimize loss of discriminator assuming label is True
		y.data.resize_(current_batch_size).fill_(1)
		y_pred_fake = D(x_fake)
		errG = criterion(y_pred_fake, y)
		errG.backward(retain_graph=True)
		D_G = y_pred_fake.data.mean()
		optimizerG.step()

		current_step = i + epoch*len(dataset)
		# Log results so we can see them in TensorBoard after
		log_value('errD', errD.data[0], current_step)
		log_value('errG', errG.data[0], current_step)

		if i % 50 == 0:
			end = time.time()
			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f time:%.4f' % (epoch, param.n_epoch, i, len(dataset), errD.data[0], errG.data[0], D_real, D_fake, D_G, end - start))
			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f time:%.4f' % (epoch, param.n_epoch, i, len(dataset), errD.data[0], errG.data[0], D_real, D_fake, D_G, end - start), file=log_output)
	# Fake images saved
	fake_test = G(z_test)
	vutils.save_image(fake_test.data, '%s/run-%d/images/fake_samples_epoch%03d.png' % (param.output_folder, run, epoch), normalize=True)

	# Save every epoch
	if epoch % 25 == 0:
		torch.save(G.state_dict(), '%s/run-%d/models/G_epoch_%d.pth' % (param.output_folder, run, epoch))
		torch.save(D.state_dict(), '%s/run-%d/models/D_epoch_%d.pth' % (param.output_folder, run, epoch))
