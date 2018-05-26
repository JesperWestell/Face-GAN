from models.mod_cdcgan import mod_cDCGAN
from models.mod2_cdcgan import mod2_cDCGAN
from models.mod3_cdcgan import mod3_cDCGAN
from models.cdcgan import cDCGAN
from models.dcgan import DCGAN

cdcgan = mod3_cDCGAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True)
cdcgan.load(checkpoint='./outputs/mod3_cdcgan_out/mod3_cdcgan_epoch_29.pth')
cdcgan.build_sample_dataset(batches=100)


