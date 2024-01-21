#!/usr/bin/env python3

import argparse
import os
import torch
from torch import nn
from torch.autograd.variable import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a GAN model for generating handwriting.")
parser.add_argument('--epochs', type=int, help='Number of epochs to train.', required=True)
parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
parser.add_argument('--output-dir', type=str, help='Directory to save output images to', required=True)
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available.')
parser.add_argument('--quiet-mode', action='store_true', help='Run script in quiet mode.')
args = parser.parse_args()

# Use GPU if it's available and if --gpu flag is passed
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

# Define hyperparameters
latent_dim = 100
img_size = 28
channels = 1  # Grayscale for handwriting
lr = 0.0002
b1 = 0.5
b2 = 0.999

# Define transformations for datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])

# Load dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


# Define Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size*img_size*channels),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img


# Define Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size*channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Noise for input to generator
def generate_noise(size):
    return Variable(torch.randn(size, latent_dim)).to(device)

# Create output dir if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Train
for epoch in range(args.epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Ground truths
        valid = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device)

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)

        #  -----------------
        #  Train Generator
        #  -----------------
        optimizer_G.zero_grad()

        z = generate_noise(imgs.size(0))

        # Generate a batch of images
        generated_imgs = generator(z)

        # Generator loss
        g_loss = adversarial_loss(discriminator(generated_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #  ---------------------
        #  Train Discriminator
        #  ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            save_image(generated_imgs.data[:25], os.path.join(args.output_dir, f"{batches_done}.png"), nrow=5, normalize=True)
            if not args.quiet_mode:
                print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Save final generator's state dict
torch.save(generator.state_dict(), os.path.join(args.output_dir, "generator.pth"))