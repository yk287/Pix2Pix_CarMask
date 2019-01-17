
'''
Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/image_folder.py
'''

import os
import os.path

extensions = [
    '.jpg', '.jpeg', '.png', 'gif',
]


def get_images(dir, extensions = extensions):
    """
    Given dir, transverse through the folders and get all the images that end in EXTENSIONS
    :param dir: root directory
    :return file_list (list): a list containing the dir + names of files as str
    """
    file_list = []

    for root, folder, files in os.walk(dir):
        for filename in files:
            if any(filename.endswith(ext) or filename.endswith(ext.upper()) for ext in extensions):
                filename = os.path.join(root, filename)
                file_list.append(filename)

    return sorted(file_list)

def get_folders(dir):

    folder_names = [x[0] for x in os.walk(dir)]

    return folder_names
