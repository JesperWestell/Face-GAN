from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.cls_gan import CLS_GAN
from models.ac_gan import AC_GAN

gan = CLS_GAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True, subset=False)
gan.load(checkpoint='./outputs/cls_gan_out/cls_gan_epoch_29.pth')
#gan = cDCGAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True, subset=False)
#gan.load(checkpoint='./outputs/cdcgan_out/cdcgan_epoch_29.pth')
gan.build_sample_dataset(batches=200)


