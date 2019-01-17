import os
import fnmatch

""""""

def get_filenames(directory, filenames):

    filelist = []

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, filenames):
            filelist.append(file)

    return filelist

#example
#d = get_filenames('./img', 'image*.png')

import imageio

def image_to_gif(directory, filenames, duration=0.5, destination=None, gifname=None):
    """
    Given a directory, filename and duration, this function creates a gif using the filename in the directory given
    with a puase of duration seconds between images

    :param directory (str): directory that holds images
    :param filename (list of str)): a list that holds str of names of filenames that will be turned into a gif
    :param duration (float): a pause between images. defaulted to 0.5 second pause
    :param destination (str): destination directory
    :param gifname (str): name for the gif file.
    :return: NA this function simply saves the gif in the directory given
    """

    if destination == None:
        destination = directory

    if gifname == None:
        gifname = 'movie'

    images = []

    for filename in filenames:
        images.append(imageio.imread(os.path.join('%s' %directory, '%s' %filename)))
    imageio.mimsave('%s%s.gif' % (directory, gifname), images, duration=duration)

#example
#image_to_gif('./img/', d)