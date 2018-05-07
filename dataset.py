import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path

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

def load_attributes(filename):
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
                          dtype=np.int, skiprows=2)
    assert files.size == data.shape[0]
    print("Finished loading {}".format(filename))
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
                 extensions = IMG_EXTENSIONS, loader=default_loader):
        self.root = root
        self.attr_file = attr_file
        self.attr_names, self.attr = load_attributes(attr_file)
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