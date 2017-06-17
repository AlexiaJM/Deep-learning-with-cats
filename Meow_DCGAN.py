# Reference 1 : https://github.com/pytorch/examples/blob/master/dcgan/main.py
# Reference 2 : https://arxiv.org/pdf/1511.06434.pdf
# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/DCGAN

## Parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64) # DCGAN paper original value used 128
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=100) # DCGAN paper original value
parser.add_argument('--h_size', type=int, default=128) # DCGAN paper original value
parser.add_argument('--lr_D', type=float, default=.00005, help='Discriminator learning rate') # 1/4 of DCGAN paper original value
parser.add_argument('--lr_G', type=float, default=.0002, help='Generator learning rate') # DCGAN paper original value
parser.add_argument('--n_epoch', type=int, default=250)
parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
parser.add_argument('--seed', type=int)
parser.add_argument('--input_folder', default='/home/alexia/Datasets/Meow', help='input folder, do not finish with a /')
parser.add_argument('--output_folder', default='/home/alexia/Output/DCGAN', help='output folder, do not finish with a /')
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

# Loading data in batch
dataset = torch.utils.data.DataLoader(data, batch_size=param.batch_size, shuffle=True)

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
			torch.nn.Sigmoid()
			# Size = 1 x 1 x 1 (Is a real cat or not?)
		)
	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
			output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
		else:
			output = self.main(input)
		# Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
		return output.view(-1, 1)

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
	############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################

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

		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################

		G.zero_grad()
		# Generator wants to fool discriminator so it wants to minimize loss of discriminator assuming label is True
		y.data.resize_(current_batch_size).fill_(1)
		y_pred_fake = D(x_fake)
		errG = criterion(y_pred_fake, y)
		errG.backward(retain_variables=True)
		D_G = y_pred_fake.data.mean()
		optimizerG.step()

		current_step = i + epoch*len(dataset)
		# Log results so we can see them in TensorBoard after
		log_value('errD', errD.data[0], current_step)
		log_value('errG', errG.data[0], current_step)

		if i % 10 == 0:
			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, param.n_epoch, i, len(dataset), errD.data[0], errG.data[0], D_real, D_fake, D_G))
			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, param.n_epoch, i, len(dataset), errD.data[0], errG.data[0], D_real, D_fake, D_G), file=log_output)
	# Fake images saved
	fake_test = G(z_test)
	vutils.save_image(fake_test.data, '%s/run-%d/images/fake_samples_epoch%03d.png' % (param.output_folder, run, epoch), normalize=True)

	# Save every epoch
	torch.save(G.state_dict(), '%s/run-%d/models/G_epoch_%d.pth' % (param.output_folder, run, epoch))
	torch.save(D.state_dict(), '%s/run-%d/models/D_epoch_%d.pth' % (param.output_folder, run, epoch))
