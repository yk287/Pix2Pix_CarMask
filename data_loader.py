#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import random

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from skimage import color

from image_folder import get_images

import torchvision.transforms.functional as TVF

class Pix2Pix_AB_Dataloader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, img_folder, target_folder, transform=None, size =256, randomcrop = 224, hflip=0.5, vflip=0.5, train=True):
        '''

        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.img_folder = img_folder
        self.size = size #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.transforms = transform
        self.train = train

        self.file_names = get_images(img_folder)
        self.target_names = get_images(target_folder)

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        left_image = Image.open(self.file_names[index]).convert('RGB')
        right_image = Image.open(self.target_names[index]).convert('RGB')


        '''
        Resize
        '''
        resize = transforms.Resize(size=(self.size, self.size))
        left_image = resize(left_image)
        right_image = resize(right_image)

        '''
        RandomCrop
        '''

        if self.train:

            i, j, h, w = transforms.RandomCrop.get_params(
                left_image, output_size=(self.randomCrop, self.randomCrop)
            )
            left_image = TVF.crop(left_image, i, j, h, w)
            right_image = TVF.crop(right_image, i, j, h, w)

            if random.random() >= self.hflip:
                left_image = TVF.hflip(left_image)
                right_image = TVF.hflip(right_image)

            if random.random() >= self.vflip:
                left_image = TVF.vflip(left_image)
                right_image = TVF.vflip(right_image)

        left_image = self.transforms(left_image)
        right_image = self.transforms(right_image)

        return left_image, right_image

def GrayScaleAndColor(image):
    '''
    RGB2LAB converter that takes in a tensor of an image and converts it to LAB image
    :param image (tensor): a tensor of an image
    :return: returns a list that contains a grayscale image and the corresponding color image
    '''

    im = np.array(image)
    im = np.rollaxis(im, 0, 3)
    #image tensor has C X Width X height dimensions
    #color.xyz2lab takes in a ndarray of width X height X C

    lab = color.rgb2lab(im).astype(np.float32)
    lab_t = transforms.ToTensor()(lab)

    #These values are straight from Jun-Yanz github
    A = lab_t[[0], ...] / 50.0 - 1.0
    B = lab_t[[1, 2], ...] / 110.0

    return [A, B]


def GrayScaleAndColor1(image):
    '''
    RGB2LAB converter that takes in a tensor of an image and converts it to LAB image
    :param image (tensor): a tensor of an image
    :return: returns a list that contains a grayscale image and the corresponding color image
    '''

    transform_to_gray = transforms.Compose([
        transforms.Grayscale(1)
    ])

    to_tensor_transoform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    gray = transform_to_gray(image)
    gray = to_tensor_transoform(gray)

    image = to_tensor_transoform(image)
    return [gray, image]