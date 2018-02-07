import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from src.data.utils import validation_split


def get_permuted_MNIST(path, batch_size):
    im_width = im_height = 28
    val_size = 10000

    rand_perm = RandomPermutation(0, 0, im_width, im_height)
    normalization = transforms.Normalize((0.1307,), (0.3081,))

    #Todo: rethink RandomPermutation usage slows down dataloading by a factor > 6, Should try directly on batches.
    transfrom = transforms.Compose([
        transforms.ToTensor(),
        rand_perm,
        normalization]
    )

    train_set = MNIST(root=path, train=True, download=True, transform=transfrom)
    test_set = MNIST(root=path, train=False, download=True, transform=transfrom)
    train_set, val_set = validation_split(train_set, transfrom, transfrom, val_size=val_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) if train_set is not None else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) if test_set is not None else None
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set is not None else None

    return train_loader, val_loader, test_loader


class RandomPermutation(object):
    """
    Applies a constant random permutation to the images.
    """

    def __init__(self, x_off=0, y_off=0, width=None, height=None):
        self.x_off = x_off
        self.y_off = y_off
        self.width = width
        self.height = height
        self.x_max = x_off + width
        self.y_max = y_off + height
        self.kernel = torch.randperm(width * height)

    def __call__(self, input):
        return rand_perm_(input, self.x_off, self.y_off, self.x_max, self.y_max, self.kernel)


def rand_perm_(img, x, y, x_max, y_max, kernel):
    """
    Applies INPLACE the random permutation defined in `kernel` to the image `img` on
    the zone defined by `x`, `y`, `x_max`, `y_max`
    :param img: Input image of dimension (C*W*H)
    :param x: offset on x axis
    :param y: offset on y axis
    :param x_max: end of the zone to permute on the x axis
    :param y_max: end of the zone to permute on the y axis
    :param kernel: LongTensor of dim 1 containing one value for each point in the zone to permute
    :return: teh permuted image (even if the permutation is done inplace).
    """
    zone = img[:, x:x_max, y:y_max].contiguous()
    img[:, x:x_max, y:y_max] = zone.view(img.size(0), -1).index_select(1, kernel).view(zone.size())
    return img
