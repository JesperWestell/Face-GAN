from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from utils.dataset import CelebADataset
from utils.utils import PrintLayer, AttributeGenerator, generate_fixed, \
    smooth_labels, mismatch_attributes, add_noise

out_folder = './outputs/ac_gan_out/'
db_folder = './databases/ac_gan/imgs/'

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


class YModule(nn.Module):
    # Simple wrapper for network with two inputs and one output,
    # hence a Y network
    def __init__(self, in1, in2, out):
        super(YModule, self).__init__()
        self.in1 = in1
        self.in2 = in2
        self.out = out

    def forward(self, input):
        input1, input2 = input
        out1 = self.in1(input1)  # Noise/image
        out2 = self.in2(input2)  # Attributes
        output = self.out(torch.cat([out1, out2], 1))
        return output

class ReverseYModule(nn.Module):
    # Simple wrapper for network with one input and two outputs,
    # hence a reversed Y network
    def __init__(self, in1, out1, out2):
        super(ReverseYModule, self).__init__()
        self.in1 = in1
        self.out1 = out1
        self.out2 = out2

    def forward(self, input):
        middle = self.in1(input)
        out1 = self.out1(middle)  # Source score
        out2 = self.out2(middle)  # Class score
        return out1, out2


class Expand(nn.Module):
    # Expands a tensor, from c, 1, 1 to c, d, d
    def __init__(self, d):
        super(Expand, self).__init__()
        self.d = d

    def forward(self, input):
        output = input.expand(-1, -1, self.d, self.d)
        return output


class Unsqueeze(nn.Module):
    # Simple wrapper for network with two inputs and one output,
    # hence a Y network
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, input):
        output = input.unsqueeze(-1).unsqueeze(-1)
        return output


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, na):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        noise_layer = nn.Sequential(
            # input is z, going into a convolution
            # state size. (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 7, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 7),
            nn.ReLU(True)
            # state size. (ngf*7) x 4 x 4
        )
        attribute_layer = nn.Sequential(
            # input is t, going into a convolution
            # state size. (na)
            Unsqueeze(),
            # state size. (na) x 1 x 1
            nn.ConvTranspose2d(na, ngf * 1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True)
            # state size. (ngf*1) x 4 x 4
        )
        output_layer = nn.Sequential(
            # input is noise_layer(z) + attribute_layer(t),
            # going into a convolution
            # state size. (ngf*7 + ngf*1) x 4 x 4
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
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.main = YModule(noise_layer, attribute_layer, output_layer)

    def forward(self, input, attribute):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main(),
                                               (input, attribute),
                                               range(self.ngpu))
        else:
            output = self.main((input, attribute))
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, na):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.na = na
        image_layer = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        source_layer = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

        class_layer = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, na, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

        self.main = ReverseYModule(image_layer, source_layer, class_layer)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output1, output2 = nn.parallel.data_parallel(self.main(),
                                               input,
                                               range(self.ngpu))
        else:
            output1, output2 = self.main(input)

        return output1.view(-1, 1).squeeze(1), output2.view(-1, self.na)


class AC_GAN():
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
                 cuda=False,
                 ngpu=1,
                 netG='',
                 netD='',
                 random_seed=None,
                 c_weight=1,
                 subset=False):
        self.nz = nz
        self.current_epoch = 0
        self.step = 0
        self.c_weight = c_weight
        if subset:
            na = 8
        else:
            na = 40

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
                                     ]),
                                     target_transform=transforms.Lambda(
                                         lambda a: torch.from_numpy(a)),
                                     subset=subset)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=workers)

        self.device = torch.device("cuda:0" if cuda else "cpu")

        self.netG = Generator(ngpu, nz, ngf, nc, na).type(self.dtype).to(
            self.device)
        self.netG.apply(weights_init)
        if netG != '':
            self.netG.load_state_dict(torch.load(netG))
        print(self.netG)

        self.netD = Discriminator(ngpu, ndf, nc, na).type(self.dtype).to(
            self.device)
        self.netD.apply(weights_init)
        if netD != '':
            self.netD.load_state_dict(torch.load(netD))
        print(self.netD)

        self.criterion = nn.BCELoss()

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr,
                                     betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr,
                                     betas=(0.5, 0.999))
        self.attribute_generator = \
            AttributeGenerator(self.dataset.get_attributes())

        # Used when plotting arbitrary faces
        self.fixed_noise = torch.randn(batch_size, nz, 1, 1,
                                       device=self.device).type(self.dtype)
        self.fixed_attributes = self.attribute_generator.sample(
            batch_size).type(self.dtype)
        # Used when plotting conditional faces
        self.gradient_noise = torch.randn(8, nz, 1, 1, device=self.device).type(
            self.dtype)
        self.gradient_noise = self.gradient_noise.repeat(1, 8, 1, 1).view(
            (8, 8, nz, 1, 1)).view(8 * 8, nz, 1, 1)
        self.gradient_attributes = generate_fixed(self.attribute_generator,
                                                  self.dataset.get_attribute_names()).type(
            self.dtype)

    def train(self, niter=25, checkpoint=None):
        if checkpoint != None:
            try:
                self.load_checkpoint(checkpoint)
            except:
                print(' [*] No checkpoint!')
                self.current_epoch = 0
                self.step = 0

        writer = SummaryWriter('./summaries/ac_gan')

        real_label = 1
        fake_label = 0

        initial_noise_strength = 0.1
        noise_anneal_epoch = 20
        initial_weight = 1
        weight_anneal_epoch = 30
        weight_start_epoch = 5

        for epoch in range(self.current_epoch, niter):
            img_weight = min(1,
                                  max(initial_weight * 0.1,
                                      initial_weight * (
                                              1 - 0.9 * (
                                                  epoch - weight_start_epoch + 1) / (
                                                          weight_anneal_epoch - weight_start_epoch))))
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize 0.5( log(Ds(x))
                #                                         + log(1-Ds(G(z,y))) )
                #                                + 0.5( log(Ds(x))
                #                                         + log(Ds(G(z,y))) )
                ###########################
                self.netD.zero_grad()
                real_img = data[0].to(self.device)
                real_img = add_noise(real_img, initial_noise_strength,
                                     noise_anneal_epoch, epoch, device=self.device)
                real_attr = data[1].to(self.device)
                real_attr_sigmoid = 0.5*real_attr+0.5
                batch_size = real_attr.size(0)
                z = torch.randn(batch_size, self.nz, 1, 1,
                                device=self.device)
                fake_img = self.netG(z, real_attr)
                fake_img = add_noise(fake_img, initial_noise_strength,
                                     noise_anneal_epoch, epoch, device=self.device)

                # train with real images
                label = torch.full((batch_size,), real_label,
                                   device=self.device)
                r_output_s, r_output_c = self.netD(real_img)
                errD_real_img = img_weight*self.criterion(r_output_s, label) + \
                                (1-img_weight)*self.criterion(r_output_c, real_attr_sigmoid)
                errD_real_img.backward()
                D_x_s = r_output_s.mean().item()
                D_x_c = r_output_c.mean().item()

                # train with fake images
                f_output_s, f_output_c = self.netD(fake_img)
                errD_fake_img = (1-img_weight)*self.criterion(f_output_c, real_attr_sigmoid)
                label.fill_(fake_label)
                errD_fake_img += img_weight*self.criterion(f_output_s, label)
                errD_fake_img.backward(retain_graph=True)
                D_G_z_s = f_output_s.mean().item()
                D_G_z_c = f_output_c.mean().item()

                # Compute total loss for D and optimize
                errD = errD_real_img + errD_fake_img
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)
                errG = img_weight*self.criterion(f_output_s, label) + \
                       (1-img_weight)*self.criterion(f_output_c, real_attr_sigmoid)
                errG.backward()
                self.optimizerG.step()

                print(
                    '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Ds(x): %.4f Ds(x^): %.4f Dc(x) %.4f Dc(x^): %.4f'
                    % (epoch, niter, i, len(self.dataloader),

                       errD.item(), errG.item(), D_x_s, D_G_z_s, D_x_c, D_G_z_c))

                writer.add_scalars('D/loss',
                                   {'D loss': errD.item()},
                                   global_step=self.step)
                writer.add_scalars('D/D_s(x)',
                                   {'D_s(x)': D_x_s,
                                    'D_c(x)': D_x_c,
                                    'D_c(x^)': D_G_z_c},
                                   global_step=self.step)
                writer.add_scalars('G/loss',
                                   {'G loss': errG.item()},
                                   global_step=self.step)
                writer.add_scalars('D/D(G(z))',
                                   {'D_s(x^)': D_G_z_s},
                                   global_step=self.step)
                self.step += 1
                if i % 100 == 0:
                    vutils.save_image(real_img[:64],
                                      '%s/real_samples.png' % out_folder,
                                      normalize=True)
                    fake = self.netG(self.fixed_noise[:64],
                                     self.fixed_attributes[:64])
                    vutils.save_image(fake.detach(),
                                      '%s/fake_samples_epoch_%03d.png' % (
                                          out_folder, epoch),
                                      normalize=True)
                    fake = self.netG(self.gradient_noise[:64],
                                     self.gradient_attributes[:64])
                    vutils.save_image(fake.detach(),
                                      '%s/gradient_samples_epoch_%03d.png' % (
                                          out_folder, epoch),
                                      normalize=True)

            # do checkpointing
            self.current_epoch = epoch
            self.save_checkpoint('%s/ac_gan_epoch_%d.pth' % (out_folder, epoch))

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

    def load_and_sample(self, checkpoint=None, save_path=out_folder + 'test'):
        self.load(checkpoint)
        fake = self.netG(self.fixed_noise[:64], self.fixed_attributes[:64])
        vutils.save_image(fake, save_path + '_sample.png',
                          normalize=True)
        fake = self.netG(self.gradient_noise[:64],
                         self.gradient_attributes[:64])
        vutils.save_image(fake, save_path + '_gradient.png',
                          normalize=True)

    def build_sample_dataset(self, batches=10):
        # Samples from the generator and builds a dataset from the samples, storing it in a folder
        batch_size = 64
        for b in range(batches):
            noise = torch.randn(batch_size, self.nz, 1, 1,
                                device=self.device).type(self.dtype)
            attributes = self.attribute_generator.sample(batch_size).type(
                self.dtype)
            fake = self.netG(noise, attributes)
            for i in range(batch_size):
                f = fake[i, :, :, :]
                vutils.save_image(f, db_folder + 'sample_%d.png' % (
                            i + b * batch_size),
                                  normalize=True)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.current_epoch = state['epoch'] + 1  # Start on next epoch
        self.step = state['step']
        self.netD.load_state_dict(state['netD'])
        self.netG.load_state_dict(state['netG'])
        self.optimizerD.load_state_dict(state['optimizerD'])
        self.optimizerG.load_state_dict(state['optimizerG'])
        print('model loaded from %s' % checkpoint_path)

    def save_checkpoint(self, checkpoint_path):
        state = {'epoch': self.current_epoch,
                 'step': self.step,
                 'netD': self.netD.state_dict(),
                 'netG': self.netG.state_dict(),
                 'optimizerD': self.optimizerD.state_dict(),
                 'optimizerG': self.optimizerG.state_dict()}
        torch.save(state, checkpoint_path)
        print('model saved to %s' % checkpoint_path)
