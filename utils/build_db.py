#from mod_cdcgan import mod_cDCGAN
#from cdcgan import cDCGAN
from dcgan import DCGAN

cdcgan = DCGAN('../data/resized_celebA/', '../data/Anno/list_attr_celeba.txt', cuda=True)
cdcgan.load(checkpoint='./dcgan_out/dcgan_epoch_24.pth')
cdcgan.build_sample_dataset(batches=100)


