# Reference 1 : https://github.com/pytorch/examples
# Reference 3 : https://arxiv.org/pdf/1511.06434.pdf
# Reference 4 : https://arxiv.org/pdf/1701.07875.pdf
# Reference 5 : https://github.com/martinarjovsky/WassersteinGAN
# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/WGAN

### Doesn't seem to converge yet :(

## Parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=100) # DCGAN paper original value
parser.add_argument('--h_size', type=int, default=128) # DCGAN paper original value
parser.add_argument('--lr_D', type=float, default=.00001, help='Discriminator learning rate') # WGAN original value
parser.add_argument('--lr_G', type=float, default=.00001, help='Generator learning rate')
parser.add_argument('--n_iter', type=int, default=100000, help='Number of iterations')
parser.add_argument('--n_critic', type=int, default=5, help='Number of training with D before training G') # WGAN original value
parser.add_argument('--clip', type=float, default=.01, help='Clipping value') # WGAN original value
parser.add_argument('--seed', type=int)
parser.add_argument('--input_folder', default='/home/alexia/Datasets/Meow', help='input folder, do not finish with a /')
parser.add_argument('--output_folder', default='/home/alexia/Output/WGAN', help='output folder, do not finish with a /')
parser.add_argument('--G_load', default='', help='Full path to Generator model to load (ex: /home/output_folder/run-5/models/G_epoch_11.pth)')
parser.add_argument('--D_load', default='', help='Full path to Discriminator model to load (ex: /home/output_folder/run-5/models/D_epoch_11.pth)')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
param = parser.parse_args()

## Imports

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

import numpy
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

# To see images
from IPython.display import Image
to_img = transf.ToPILImage()

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

# Generate a random sample
def generate_random_sample():
	while True:
		random_indexes = numpy.random.choice(data.__len__(), size=param.batch_size, replace=False)
		batch = [data[i][0] for i in random_indexes]
		yield torch.stack(batch, 0)
random_sample = generate_random_sample()

## Models

# DCGAN generator
class _G(torch.nn.Module):
	def __init__(self):
		super(_G, self).__init__()
		self.main = torch.nn.Sequential(
			# Z_size random numbers
			torch.nn.ConvTranspose2d(param.z_size, param.h_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
			torch.nn.BatchNorm2d(param.h_size * 8),
			torch.nn.ReLU(),
			# Size = (H_size * 8) x 4 x 4
			torch.nn.ConvTranspose2d(param.h_size * 8, param.h_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.BatchNorm2d(param.h_size * 4),
			torch.nn.ReLU(),
			# Size = (H_size * 4) x 8 x 8
			torch.nn.ConvTranspose2d(param.h_size * 4, param.h_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.BatchNorm2d(param.h_size * 2),
			torch.nn.ReLU(),
			# Size = (H_size * 2) x 16 x 16
			torch.nn.ConvTranspose2d(param.h_size * 2, param.h_size, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.BatchNorm2d(param.h_size),
			torch.nn.ReLU(),
			# Size = H_size x 32 x 32
			torch.nn.ConvTranspose2d(param.h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.Tanh()
			# Size = n_colors x 64 x 64
		)
	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
		else:
			output = self.main(input)
		return output

# DCGAN discriminator (using somewhat the reverse of the generator)
class _D(torch.nn.Module):
	def __init__(self):
		super(_D, self).__init__()
		self.main = torch.nn.Sequential(
			# Size = n_colors x 64 x 64
			torch.nn.Conv2d(param.n_colors, param.h_size, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.LeakyReLU(0.2, inplace=True),
			# Size = H_size x 32 x 32
			torch.nn.Conv2d(param.h_size, param.h_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.BatchNorm2d(param.h_size * 2),
			torch.nn.LeakyReLU(0.2, inplace=True),
			# Size = (H_size * 2) x 16 x 16
			torch.nn.Conv2d(param.h_size * 2, param.h_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.BatchNorm2d(param.h_size * 4),
			torch.nn.LeakyReLU(0.2, inplace=True),
			# Size = (H_size * 4) x 8 x 8
			torch.nn.Conv2d(param.h_size * 4, param.h_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
			torch.nn.BatchNorm2d(param.h_size * 8),
			torch.nn.LeakyReLU(0.2, inplace=True),
			# Size = (H_size * 8) x 4 x 4
			torch.nn.Conv2d(param.h_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
			# Note: No more sigmoid in WGAN, we take the mean now
			# Size = 1 x 1 x 1 (Is a real cat or not?)
		)
	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
		else:
			output = self.main(input)
		# From batch_size x 1 x 1 to 1 x 1 x 1 by taking the mean (DCGAN used the sigmoid instead before)
		output = output.mean(0)
		# Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
		return output.view(1)

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
G = _G()
D = _D()

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

# Soon to be variables
x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
# This is to see during training, size and values won't change
z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1)

# Everything cuda
if param.cuda:
	G = G.cuda()
	D = D.cuda()
	x = x.cuda()
	z = z.cuda()
	z_test = z_test.cuda()

# Now Variables
x = Variable(x)
z = Variable(z)
z_test = Variable(z_test)

# Optimizer
optimizerD = torch.optim.RMSprop(D.parameters(), lr=param.lr_D)
optimizerG = torch.optim.RMSprop(G.parameters(), lr=param.lr_G)


## Fitting model
for i in range(param.n_iter):

	# Wassertein distance for plot
	W_distance = 0

	# Hack used by original paper
	if i < 25 or i % 500 == 0:
		N_critic = 100
	else:
		N_critic = param.n_critic

	for t in range(N_critic):
		########################
		# (1) Update D network #
		########################

		D.zero_grad()

		# Sample real data
		real_images = random_sample.__next__()
		if param.cuda:
			real_images = real_images.cuda()
		x.data.copy_(real_images)
		# Discriminator Loss real
		y_pred = D(x)
		errD_real = y_pred.mean()
		errD_real.backward()

		# Sample fake data
		z.data.normal_(0, 1)
		x_fake = G(z)
		# Discriminator Loss fake
		y_pred_fake = D(x_fake.detach())
		errD_fake = y_pred_fake.mean()
		errD_fake.backward()

		# Optimize
		errD = -(errD_real - errD_fake)
		optimizerD.step()

		# Clip weights
		for p in D.parameters():
			p.data.clamp_(-param.clip, param.clip)

		# Wassertein distance for plot
		W_distance = W_distance + errD.data[0]

	#########################
	# (2) Update G network: #
	#########################

	G.zero_grad()

	# Sample fake data
	z.data.normal_(0, 1)
	x_fake = G(z)
	# Generator Loss
	y_pred_fake = D(x_fake)
	errG = -y_pred_fake.mean()
	errG.backward()
	optimizerG.step()

	# Log results so we can see them in TensorBoard after
	log_value('errD', errD.data[0], i)
	log_value('errG', errG.data[0], i)

	if i % 1 == 0:
		print('[i=%d] W_distance: %.4f Loss_G: %.4f' % (i, W_distance, errG.data[0]))
		print('[i=%d] W_distance: %.4f Loss_G: %.4f' % (i, W_distance, errG.data[0]), file=log_output)
		# Fake images saved
		fake_test = G(z_test)
		vutils.save_image(fake_test.data, '%s/run-%d/images/fake_samples_iter%05d.png' % (param.output_folder, run, i), normalize=True)
		#vutils.save_image(x.data, '%s/run-%d/images/real_samples_iter%05d.png' % (param.output_folder, run, i/25), normalize=True)
	# Save models
	if i % 100 == 0:
		torch.save(G.state_dict(), '%s/run-%d/models/G_%d.pth' % (param.output_folder, run, i))
		torch.save(D.state_dict(), '%s/run-%d/models/D_%d.pth' % (param.output_folder, run, i))
