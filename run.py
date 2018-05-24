import sys
print(sys.path)

from models.mod_cdcgan import mod_cDCGAN
from models.cdcgan import cDCGAN
from models.dcgan import DCGAN
from models.mod2_cdcgan import mod2_cDCGAN


image_folder = '../data/resized_celebA/'
attribute_folder = '../data/Anno/list_attr_celeba.txt'

mod2_cdcgan = mod2_cDCGAN(image_folder, attribute_folder, cuda=False)
#mod_cdcgan.load(checkpoint='./mod_cdcgan_out/mod_cdcgan_epoch_24.pth')
#mod_cdcgan.build_sample_dataset(batches=1)
mod2_cdcgan.train(1)
#mod_cdcgan.load_and_sample(checkpoint='./outputs/mod_cdcgan_out/mod_cdcgan_epoch_23.pth')
