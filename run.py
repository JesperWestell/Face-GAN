from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.cls_gan import CLS_GAN
from models.ac_gan import AC_GAN
from models.new_cls_gan import new_CLS_GAN


image_folder = '../data/resized_celebA/'
attribute_folder = '../data/Anno/list_attr_celeba.txt'

gan = CLS_GAN(image_folder, attribute_folder, cuda=True, c_weight=4, subset=False)
gan.train(30)

#gan = cDCGAN(image_folder, attribute_folder, cuda=True, subset=False)
#gan.train(30)

#gan = CLS_GAN(image_folder, attribute_folder, cuda=True, subset=False)
#gan.load(checkpoint='./outputs/new_cls_gan_out/new_cls_gan_epoch_15.pth')
#gan.build_sample_dataset(batches=1)
#gan.train(30)
#gan.load_and_sample(checkpoint='./outputs/mod_cdcgan_out/mod_cdcgan_epoch_23.pth')


