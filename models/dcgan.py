from __future__ import print_function
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
from tensorboardX import SummaryWriter
from dataset import CelebADataset

out_folder = './outputs/dcgan_out/'
db_folder = './dcgan_database/imgs/'
try:
    os.makedirs(out_folder)
except OSError:
    pass
try:
    os.makedirs(db_folder)
except OSError:
    pass

def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class DCGAN():
    def __init__(self,
                 dataroot,
                 attr_file,
                 lr=2e-4,
                 workers=5,
                 batch_size=64,
                 nz=100,
                 ngf=64,
                 ndf=64,
                 nc=3,
                 cuda = False,
                 ngpu=1,
                 netG='',
                 netD='',
                 random_seed=None):
        self.nz = nz
        self.current_epoch = 0;

        if random_seed is None:
            random_seed = random.randint(1, 10000)
        print("Random Seed: ", random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        cudnn.benchmark = True

        if torch.cuda.is_available() and not cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")

        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.dataset = CelebADataset(root=dataroot,
                        attr_file=attr_file,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0, 0, 0),
                                                 (1, 1, 1)),
                        ]))

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=workers)

        self.device = torch.device("cuda:0" if cuda else "cpu")

        self.netG = Generator(ngpu, nz, ngf, nc).to(self.device)
        self.netG.apply(weights_init)
        if netG != '':
            self.netG.load_state_dict(torch.load(netG))
        print(self.netG)

        self.netD = Discriminator(ngpu, ndf, nc).to(self.device)
        self.netD.apply(weights_init)
        if netD != '':
            self.netD.load_state_dict(torch.load(netD))
        print(self.netD)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(batch_size, nz, 1, 1, device=self.device)

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr,
                                betas=(0.9, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr,
                                betas=(0.9, 0.999))

    def train(self, niter=25, checkpoint = None):
        if checkpoint != None:
            try:
                self.load_checkpoint(checkpoint)
            except:
                print(' [*] No checkpoint!')
                self.current_epoch = 0

        writer = SummaryWriter('./summaries/dcgan')

        real_label = 1
        fake_label = 0

        for epoch in range(self.current_epoch, self.current_epoch+niter):
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.netD.zero_grad()
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=self.device)

                output = self.netD(real_cpu)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(fake_label)
                output = self.netD(fake.detach())
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(
                    real_label)  # fake labels are real for generator cost
                output = self.netD(fake)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                print(
                    '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, niter, i, len(self.dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                step = epoch * len(self.dataset) + i
                writer.add_scalars('D/loss',
                                   {'D loss': errD.item()},
                                   global_step=step)
                writer.add_scalars('D/D(x)',
                                   {'D(x)': D_x},
                                   global_step=step)
                writer.add_scalars('G/loss',
                                   {'G loss': errG.item()},
                                   global_step=step)
                writer.add_scalars('D/D(G(z))',
                                   {'D_G_z1': D_G_z1, 'D_G_z2': D_G_z2},
                                   global_step=step)
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % out_folder,
                                      normalize=True)
                    fake = self.netG(self.fixed_noise)
                    vutils.save_image(fake.detach(),
                                      '%s/fake_samples_epoch_%03d.png' % (
                                      out_folder, epoch),
                                      normalize=True)

            # do checkpointing
            self.save_checkpoint('%s/dcgan_epoch_%d.pth' % (out_folder, epoch))

    def sample(self, num_samples):
        noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
        imgs = self.netG(noise)
        vutils.save_image(imgs.detach(),
                          '%s/samples.png' % out_folder,
                          normalize=True)

    def load(self, checkpoint=None):
        if checkpoint != None:
            try:
                self.load_checkpoint(checkpoint)
            except:
                print(' [*] No checkpoint!')
                self.current_epoch = 0

    def load_and_sample(self, checkpoint=None, save_path=out_folder+'test'):
        self.load(checkpoint)
        fake = self.netG(self.fixed_noise[:64])
        vutils.save_image(fake,save_path+'_sample.png',
                          normalize=True)

    def build_sample_dataset(self, batches=10):
        # Samples from the generator and builds a dataset from the samples, storing it in a folder
        batch_size = 64
        for b in range(batches):
            noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device).type(self.dtype)
            fake = self.netG(noise)
            for i in range(batch_size):
                f = fake[i,:,:,:]
                vutils.save_image(f, db_folder + 'sample_%d.png'% (i+b*batch_size),
                                  normalize=True)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.current_epoch = state['epoch']
        self.netD.load_state_dict(state['netD'])
        self.netG.load_state_dict(state['netG'])
        self.optimizerD.load_state_dict(state['optimizerD'])
        self.optimizerG.load_state_dict(state['optimizerG'])
        print('model loaded from %s' % checkpoint_path)

    def save_checkpoint(self, checkpoint_path):
        state = {'epoch': self.current_epoch,
                 'netD': self.netD.state_dict(),
                 'netG': self.netG.state_dict(),
                 'optimizerD': self.optimizerD.state_dict(),
                 'optimizerG': self.optimizerG.state_dict()}
        torch.save(state, checkpoint_path)
        print('model saved to %s' % checkpoint_path)
