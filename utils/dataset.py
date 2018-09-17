import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def plot_heat_map(d, names):
    names = [ n.replace('_', ' ') for n in names]
    corr = np.corrcoef(d)
    # This dictionary defines the colormap
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.5, 0.5, 0.5),
                     (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                     (0.5, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),

             'blue': ((0.0, 1.0, 1.0),
                      (0.5, 0.5, 0.5),
                      (1.0, 0.0, 0.0))
             }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

    # Make a figure and axes

    fig, ax = plt.subplots(1, dpi=200)

    # Plot the fake data
    p = ax.pcolormesh(corr, cmap=GnRd, vmin=-0.8, vmax=1)

    plt.xticks([x-0.5 for x in range(1, 41)],
               names,
               rotation='vertical', size= 'xx-small')
    plt.yticks([x-0.5 for x in range(1, 41)],
               names,
               size='xx-small')
    #ax.get_xaxis().set_ticks(names)
    #ax.get_yaxis().set_ticks(names)

    # Make a colorbar
    fig.colorbar(p, ax=ax)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('corr.png', bbox_inches='tight')
    plt.show()

def load_attributes(filename, subset):
    # kyamagu/build_celeba_lmdb.py
    with open(filename, 'r') as f:
        num_images = int(f.readline().strip())
        print("{} records".format(num_images))
        attributes = f.readline().strip().split(" ")
        print("{} attributes".format(len(attributes)))
        print("{}".format(attributes))
        files = np.loadtxt(f, usecols=[0], dtype=np.str)
        f.seek(0)
        data = np.loadtxt(f, usecols=[i + 1 for i in range(len(attributes))],
                          dtype=np.float32, skiprows=2)
        plot_heat_map(data.T, attributes)
    assert files.size == data.shape[0]
    print("Finished loading {}".format(filename))
    if subset:
        print('Only taking subset of attributes!')
        sub_attributes = ['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Male',
                      'No_Beard', 'Smiling', 'Wearing_Hat', 'Young']
        print("{} sub attributes".format(len(sub_attributes)))
        print("{}".format(sub_attributes))
        indices = np.zeros(len(sub_attributes), dtype=np.int32)
        for i, attr in enumerate(sub_attributes):
            idx = attributes.index(attr)
            indices[i] = idx
        print("indices: {}".format(indices))
        print('whole data shape {}'.format(data.shape))
        sub_data = data[:,indices]
        print('sub data shape {}'.format(sub_data.shape))
        data = sub_data
        attributes = sub_attributes
    return attributes, data


class CelebADataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, attr_file, transform=None, target_transform=None,
                 extensions = IMG_EXTENSIONS, loader=default_loader, subset=False):
        self.root = root
        self.attr_file = attr_file
        self.attr_names, self.attr = load_attributes(attr_file, subset)
        self.imgs = make_dataset(root, extensions)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). Target is the attribute vector.
        """
        path = self.imgs[index]
        target = self.attr[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def get_attributes(self):
        return self.attr

    def get_attribute_names(self):
        return self.attr_names

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Target Location: {}\n'.format(self.attr_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
