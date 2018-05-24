import os
import sys
import random
import matplotlib.pyplot as plt
from scipy.misc import imresize

def main():
    random.seed(3)
    root = os.path.dirname(sys.argv[0])

    # root path depends on your computer
    load_root = root + '/../data/celebA/'
    save = '/../data/resized_celebA/imgs'
    save_root = root + save
    try:
        os.makedirs('.' + save + '/')
    except OSError:
        pass

    resize_size = 64

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    img_list = os.listdir(load_root)

    for i in range(len(img_list)):
        img = plt.imread(load_root + img_list[i])
        img = imresize(img, (resize_size+16, resize_size+16))[8:resize_size+8,8:resize_size+8]
        plt.imsave(fname=save_root + img_list[i], arr=img)

        if (i % 1000) == 0:
            print('%d images complete' % i)

    print('done')

main()