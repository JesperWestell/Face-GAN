from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.cls_gan import CLS_GAN
from models.ac_gan import AC_GAN


image_folder = '../data/resized_celebA/'
attribute_folder = '../data/Anno/list_attr_celeba.txt'

gan = AC_GAN(image_folder, attribute_folder, cuda=True)
#gan.load(checkpoint='./outputs/cls_gan_out/cls_gan_epoch_24.pth')
#gan.build_sample_dataset(batches=1)
gan.train(30)
#gan.load_and_sample(checkpoint='./outputs/mod_cdcgan_out/mod_cdcgan_epoch_23.pth')

gan = CLS_GAN(image_folder, attribute_folder, cuda=True)

gan.train(30)
