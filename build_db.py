from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.cls_gan import CLS_GAN
from models.ac_gan import AC_GAN

#gan = CLS_GAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True, subset=True)
#gan.load(checkpoint='./outputs/cls_gan_out/cls_gan_epoch_29.pth')
gan = AC_GAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True, subset=True)
gan.load(checkpoint='./outputs/ac_gan_out/ac_gan_epoch_29.pth')
gan.build_sample_dataset(batches=200)


