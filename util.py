
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import os


from torch.distributions import Normal

import matplotlib.ticker as ticker

from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def confusion_plot(matrix, y_category):
    '''
    A function that plots a confusion matrix
    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def accuracy(net, loader):
    '''
    A function that returns total number of correct predictions and total comparisons
    given a neural net and a pytorch data loader

    :param net: neural net
    :param loader: data loader
    :return:
    '''

    correct = 0
    total = 0

    for images, labels in iter(loader):

        output = net.forward(images)

        _, prediction = torch.max(output.data, 1)

        total += labels.shape[0] #accumulate by batch_size
        correct += (prediction == labels).sum() #accumulate by total_correct

    return correct, total

def prediction_accuracy(net, images, labels):
    '''
    A function that returns total number of correct predictions and total comparisons
    given a neural net and a pytorch data loader

    :param net: neural net
    :param loader: data loader
    :return:
    '''

    output = net.forward(images)

    _, prediction = torch.max(output.data, 1)

    total = labels.shape[0] #accumulate by batch_size
    correct = (prediction == labels).sum() #accumulate by total_correct

    return correct, total


def pred_plotter(original_image, prediction, y_label):
    '''
    Used for visual inspection of how well the classifier works on an image.

    Takes in the original_image, softmax class predictions for the image, and y_labels to plot a side by side graph that
    shows what the model predicted, and what the model looks like.

    :param original_image (tensor): a tensor that holds the values pixel values for the image
    :param prediction (tensor): a tensor that holds the softmax class probabilities for original_image
    :param y_label (list): a list that holds names for the classes.
    :return:
    '''

    fig, (ax1, ax2) = plt.subplots(figsize=(9,6), ncols=2)

    y_values = np.arange(len(y_label))

    ax1.barh(y_values, prediction.squeeze().numpy(), align = 'center')
    ax1.set_yticks(y_values)
    ax1.set_yticklabels(y_label) #use the name of the classes as labels
    ax1.invert_yaxis()

    #values for the axis and the title
    ax1.set_xlabel('Probability')
    ax1.set_title('Class Probability')

    #shows the original_image
    ax2.imshow(original_image.view(1, 28, 28).squeeze().numpy())

    plt.show()

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751

    :param input: PyTorch Tensor of shape (N, )
    :param target: PyTorch Tensor of shape (N, ). An indicator variable that is 0 or 1
    :return:
    """

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()

    return loss.mean()

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for Vanilla GANs

    :param logits_real: PyTorch Tensor of shape(N, ). Gives scores for the real data
    :param logits_fake: PyTorch Tensor of shape(N, ). Gives scores for the fake data
    :return: PyTorch Tensor containing the loss for the discriminator
    """

    labels = torch.ones(logits_real.size()).to(device) #label used to indicate whether it's real or not

    loss_real = nn.MSELoss()(logits_real, labels) #real data
    loss_fake = nn.MSELoss()(logits_fake, 1 - labels) #fake data

    loss = loss_real + loss_fake

    return loss.to(device)

def generator_loss(logits_fake):
    """
    Computes the generator loss

    :param logits_fake: PyTorch Tensor of shape (N, ). Gives scores for the real data
    :return: PyTorch tensor containing the loss for the generator
    """

    labels = torch.ones(logits_fake.size()).to(device)

    loss = bce_loss(logits_fake, labels)

    return loss.to(device)


def sample_from_prior(bottle_neck):
    """
    Samples from an N dimensional Standard Normal distribution which will be used as a prior
    :param bottle_neck : Dimension of the bottleneck in the encoder.
    :return:
    """

    Samples = Normal(torch.zeros(bottle_neck), torch.ones(bottle_neck)).sample()

    return Samples


def plotter(env_name, num_episodes, rewards_list, ylim):
    '''
    Used to plot the average over time
    :param env_name:
    :param num_episodes:
    :param rewards_list:
    :param ylim:
    :return:
    '''
    x = np.arange(0, num_episodes)
    y = np.asarray(rewards_list)
    plt.plot(x, y)
    plt.ylim(top=ylim + 10)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Avg Rewards Last 100 Episodes")
    plt.title("Rewards Over Time For %s" %env_name)
    plt.savefig("progress.png")
    plt.close()

def raw_score_plotter(scores):
    '''
    used to plot the raw score
    :param scores:
    :return:
    '''
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Train Loss')
    plt.xlabel('Number of Iterations')
    plt.title("Loss Over Time")
    plt.savefig("Train_Loss.png")
    plt.close()



def save_images_to_directory(image_tensor, directory, filename):
    directory = directory
    image = image_tensor.cpu().data

    save_name = os.path.join('%s' % directory, '%s' % filename)
    save_image(image, save_name)

    return filename