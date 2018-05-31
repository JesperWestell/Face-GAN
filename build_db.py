from models.mod_cdcgan import mod_cDCGAN
from models.mod2_cdcgan import mod2_cDCGAN
from models.mod3_cdcgan import mod3_cDCGAN
from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.cls_gan import CLS_GAN

gan = CLS_GAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True)
gan.load(checkpoint='./outputs/cls_gan_out/cls_gan_epoch_29.pth')
gan.build_sample_dataset(batches=200)


