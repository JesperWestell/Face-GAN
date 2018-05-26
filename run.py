from models.mod_cdcgan import mod_cDCGAN
from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.mod2_cdcgan import mod2_cDCGAN
from models.mod3_cdcgan import mod3_cDCGAN


image_folder = '../data/resized_celebA/'
attribute_folder = '../data/Anno/list_attr_celeba.txt'

mod3_cdcgan = mod3_cDCGAN(image_folder, attribute_folder, cuda=True)
#mod_cdcgan.load(checkpoint='./outputs/mod_cdcgan_out/mod_cdcgan_epoch_34.pth')
#mod_cdcgan.build_sample_dataset(batches=1)
mod3_cdcgan.train(30)
#mod_cdcgan.load_and_sample(checkpoint='./outputs/mod_cdcgan_out/mod_cdcgan_epoch_23.pth')
