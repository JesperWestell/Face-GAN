from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.cls_gan import CLS_GAN
from models.ac_gan import AC_GAN
from models.new_cls_gan import new_CLS_GAN


image_folder = '../data/resized_celebA/'
attribute_folder = '../data/Anno/list_attr_celeba.txt'

gan = AC_GAN(image_folder, attribute_folder, cuda=False, c_weight=4, subset=True)
#gan.train(30)
#for i in range(10):
 #   gan.load_and_sample(checkpoint='./outputs/ac_gan_out_sub/ac_gan_epoch_29.pth',
  #                  save_path='./outputs/ac_gan_out_sub/test{}'.format(i+1))

#gan = cDCGAN(image_folder, attribute_folder, cuda=True, subset=False)

#gan.train(30)

#gan = CLS_GAN(image_folder, attribute_folder, cuda=True, subset=True)
#gan.load(checkpoint='./outputs/new_cls_gan_out/new_cls_gan_epoch_15.pth')
#gan.build_sample_dataset(batches=1)
#gan.train(30)
#gan.load_and_sample(checkpoint='./outputs/mod_cdcgan_out/mod_cdcgan_epoch_23.pth')


