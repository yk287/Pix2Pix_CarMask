
import util
import os
import numpy as np
from image_to_gif import image_to_gif

import torch
import torch.nn as nn

from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=50, lambd=1):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every
        self.lambd = lambd

    def train(self):
        steps = 0

        #used to make gifs later
        pred_image_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []


        for e in range(self.epochs):
            running_loss = 0

            for input_image, target_image in iter(self.trainloader):
                steps += 1

                input_image = input_image.to(device)
                target_image = target_image.to(device)
                '''Regularization Phase (Discriminator)'''

                self.optimizer[1].zero_grad()

                image_pred = self.model[0](input_image)

                fake_logits = self.model[1](image_pred.detach())
                real_logits = self.model[1](target_image)

                discrim_loss = util.discriminator_loss(real_logits, fake_logits) * 0.5
                discrim_loss.backward()
                self.optimizer[1].step()
                '''Regularization Phase (Discriminator)'''

                '''--------Reconstruction Phase--------'''
                self.optimizer[0].zero_grad()

                gen_logits = self.model[1](image_pred)
                gen_loss = util.generator_loss(gen_logits)

                if self.lambd > 0:
                    '''If lambda is 0 then we only use cGAN'''
                    image_pred = self.model[0](input_image)
                    recon_loss = self.criterion(image_pred, target_image)

                loss = gen_loss + self.lambd * recon_loss
                '''Unlike the regular VAE, this does not impose regularization on the Latent Vector'''
                '''Only has the reconstruction error comparing the input to the output'''
                loss.backward()
                self.optimizer[0].step()
                '''--------Reconstruction Phase--------'''

                running_loss += loss.item()
                train_loss.append(loss.item())

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "LossL {:.4f}".format(running_loss))

                running_loss = 0

            if e % 1 == 0:

                pred = image_pred.cpu().data
                directory = './img/'
                filename = 'pred_image_%s.png' % e
                pred_image_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(pred, filename)

                target = target_image.cpu().data
                directory = './img/'
                filename = 'real_image%s.png' % e
                target_image_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(target, filename)

                input = input_image.cpu().data
                directory = './img/'
                filename = 'Input_Image_%s.png' % e
                input_image_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(input, filename)

                torch.save(self.model[0].state_dict(), './model/pix2pix_G.pth')
                torch.save(self.model[1].state_dict(), './model/pix2pix_D.pth')

        image_to_gif('./img/', pred_image_list, duration=1, gifname='pred')
        image_to_gif('./img/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/', input_image_list, duration=1, gifname='input')

        util.raw_score_plotter(train_loss)

