from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Root directory for dataset
dataroot = "data/WCE"

# Number of workers for dataloader
workers = 2

# Batch size during training 128
batch_size = 128


# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
                       
# creat the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	shuffle=True, num_workers=workers)
#batch_size=batch_size

# device which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
			 padding=2, normalize=True).cpu(),(1,2,0)))


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.block1 = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf * 8),
          nn.ReLU(True),
        )
        #4x4x1024
        self.block2 = nn.Sequential(
          nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf * 4),
          nn.ReLU(True),
          #nn.Upsample(scale_factor=2, mode='nearest'),
        )
        #8x8x512
        self.block3 = nn.Sequential(
          nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf * 2),
          nn.ReLU(True),
          #nn.Upsample(scale_factor=2, mode='nearest'),
        )
        #16x16x256
        self.block4 = nn.Sequential(
          nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
          #nn.Upsample(scale_factor=2, mode='nearest'),
        )
        #32x32x128
        self.block5 = nn.Sequential(
          nn.ConvTranspose2d( ngf, 64, 4, 2, 1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
        )
        #64x64x64
        self.block6 = nn.Sequential(
          nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
          nn.BatchNorm2d(32),
          nn.ReLU(True),
        )
        #128x128x32
        self.block7 = nn.Sequential(
          nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
          nn.Tanh()
        )
        #256x256x3
        
    def forward(self, input):
        input = self.block1(input)
        input = self.block2(input)
        input = self.block3(input)
        input = self.block4(input)
        input = self.block5(input)
        #z = input
        #z = self.avg(z)
        input = self.block6(input)
        
        #input = self.concat(z, input)
        #input = torch.cat((input, z),1)

        #SELayer注意力模块
        #b,c,_,_ = input.size()
        #y = self.avg_pool(input).view(b,c)
        #y = self.SELayer(y).view(b, c, 1, 1)
        #input = input * y

        input = self.block7(input)
        return input

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.block1 = nn.Sequential(
            nn.Conv2d(nc, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        #128x128x16
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            #nn.Dropout(0.5),
            )
        #64x64x32
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            #nn.Dropout(0.5),
            )
        #32x32x64
        self.block4 = nn.Sequential(
            nn.Conv2d(64, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            #nn.Dropout(0.5),
            )
        #16x16x128
        self.block5 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            #nn.Dropout(0.5),
            )
        #8x8x256
        self.block6 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            #nn.Dropout(0.5),
            )  
        #4x4x512  
        self.block7 = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )
        self.SELayer = nn.Sequential(
            nn.Linear(512, (512//4)),
            nn.ReLU(inplace=True),
            nn.Linear((512//4), 512),
            nn.Sigmoid()
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        #self.avg = nn.AvgPool2d(2)
        #self.concat = nn.cat((A, B), 0)
   

    def forward(self, input):
        input = self.block1(input)
        input = self.block2(input)
        input = self.block3(input)
        input = self.block4(input)
        input = self.block5(input)
        #z = input
        #z = self.avg(z)
        input = self.block6(input)
        #SELayer注意力模块
        b,c,_,_ = input.size()
        y = self.avg_pool(input).view(b,c)
        y = self.SELayer(y).view(b, c, 1, 1)
        input = input * y

        #input = self.concat(z, input)
        #input = torch.cat((input, z),1)

        input = self.block7(input)
        return input

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        #print(len(real_cpu))
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device, dtype=torch.float)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        #print('##############',len(output))        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        #print('&&&&&&&&&&',len(fake.detach()))
        label.fill_(fake_label)
        #print('*************',len(label))
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())


        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

print('############',len(img_list))

plt.figure(figsize=(10,5))
fig=plt.gcf()
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()

fig.savefig(r"./images/ablation/3-DCGAN256+SE/Loss-500.png", dpi=100)

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
fig=plt.gcf()
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
fig.savefig(r"./images/ablation/3-DCGAN256+SE/Realimg.png", dpi=100)

# Plot the fake images from the last epoch 2000
plt.figure(figsize=(15,15))
fig=plt.gcf()
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
fig.savefig(r"./images/ablation/3-DCGAN256+SE/Fakeimg-SED-500.png", dpi=100)

#1000
#plt.figure(figsize=(15,15))
#fig=plt.gcf()
#plt.axis("off")
#plt.title("Fake Images")
#plt.imshow(np.transpose(img_list[999],(1,2,0)))
#fig.savefig(r"./images/Fakeimg-SED-1000.png", dpi=100)

#1200
#plt.figure(figsize=(15,15))
#fig=plt.gcf()
#plt.axis("off")
#plt.title("Fake Images")
#plt.imshow(np.transpose(img_list[1199],(1,2,0)))
#fig.savefig(r"./images/Fakeimg-SED-1200.png", dpi=100)

#1500
#plt.figure(figsize=(15,15))
#fig=plt.gcf()
#plt.axis("off")
#plt.title("Fake Images")
#plt.imshow(np.transpose(img_list[1499],(1,2,0)))
#fig.savefig(r"./images/Fakeimg-SED-1500.png", dpi=100)

#1800
#plt.figure(figsize=(15,15))
#fig=plt.gcf()
#plt.axis("off")
#plt.title("Fake Images")
#plt.imshow(np.transpose(img_list[1799],(1,2,0)))
#fig.savefig(r"./images/Fakeimg-SED-1800.png", dpi=100)
