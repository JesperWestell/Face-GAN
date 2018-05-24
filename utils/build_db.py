from models.mod_cdcgan import mod_cDCGAN
from models.cdcgan import cDCGAN
from models.dcgan import DCGAN

cdcgan = cDCGAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True)
cdcgan.load(checkpoint='./cdcgan_out/cdcgan_epoch_24.pth')
cdcgan.build_sample_dataset(batches=100)


