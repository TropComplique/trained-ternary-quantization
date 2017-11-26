import numpy as np
from PIL import Image, ImageEnhance
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


TRAIN_DIR = '/home/ubuntu/data/tiny-imagenet-200/training'
VAL_DIR = '/home/ubuntu/data/tiny-imagenet-200/validation'


"""It assumes that training image data is in the following form:
TRAIN_DIR/class4/image44.jpg
TRAIN_DIR/class4/image12.jpg
...
TRAIN_DIR/class55/image33.jpg
TRAIN_DIR/class55/image543.jpg
...
TRAIN_DIR/class1/image6.jpg
TRAIN_DIR/class1/image99.jpg
...

And the same for validation data.
"""


def get_image_folders():
    """
    Build an input pipeline for training and evaluation.
    For training data it does data augmentation.
    """

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    # intensities of enhancers
    factors = {
        0: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
        1: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        2: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        3: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
    }

    # randomly change color of an image
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        # random enhancers in random order
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image

    def rotate(image):
        degree = np.clip(np.random.normal(0.0, 15.0), -40.0, 40.0)
        return image.rotate(degree, Image.BICUBIC)

    # training data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.Lambda(rotate),
        transforms.RandomCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # for validation data
    val_transform = transforms.Compose([
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # mean and std are taken from here:
    # http://pytorch.org/docs/master/torchvision/models.html

    train_folder = ImageFolder(TRAIN_DIR, train_transform)
    val_folder = ImageFolder(VAL_DIR, val_transform)
    return train_folder, val_folder
