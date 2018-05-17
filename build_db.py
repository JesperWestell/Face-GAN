from inception_score import inception_score
from mod_cdcgan import mod_cDCGAN
from cdcgan import cDCGAN
from dcgan import DCGAN


mod_cdcgan = mod_cDCGAN('../data/resized_celebA/',
                            '../data/Anno/list_attr_celeba.txt', cuda=False)
mod_cdcgan.load(checkpoint='./mod_cdcgan_out/mod_cdcgan_epoch_24.pth')
mod_cdcgan.build_sample_dataset(batches=1)


