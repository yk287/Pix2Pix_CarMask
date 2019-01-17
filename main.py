
from data_loader import Pix2Pix_AB_Dataloader, GrayScaleAndColor1
from image_folder import get_images, get_folders

import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn

from tester import Tester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/Carvana'
folder_names = get_folders(image_dir)

train_folder = folder_names[1]
target_folder = folder_names[2]

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

resize = 256
randomCrop = 224

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = Pix2Pix_AB_Dataloader(train_folder, target_folder, transform=transform_1, size = resize, randomcrop=randomCrop)
#train_data = Pix2Pix_Dataloader(train_folder, transform=transform_train, additional_transform=GrayScaleAndColor1)

train_loader = DataLoader(train_data, batch_size=32,
                        shuffle=True, num_workers=4)

test_dir = '/home/youngwook/Downloads/Carvana_Test'
folder_names = get_folders(test_dir)

train_folder = folder_names[1]
target_folder = folder_names[1]

test_data = Pix2Pix_AB_Dataloader(train_folder, target_folder, transform=transform_1, size = randomCrop, randomcrop=randomCrop, train=False)
#train_data = Pix2Pix_Dataloader(train_folder, transform=transform_train, additional_transform=GrayScaleAndColor1)

test_loader = DataLoader(test_data, batch_size=2,
                        shuffle=True, num_workers=4)

from network import AutoEncoder_Unet, discriminator
from trainer import trainer

AE = AutoEncoder_Unet(3, 3).to(device)
D = discriminator(3, 3).to(device)
LR = 0.0002

criterion = nn.L1Loss()

unet_optim = optim.Adam(AE.parameters(), lr=LR, betas=(0.5, 0.999))
D_optim = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

optim_list = [unet_optim, D_optim]
model_list = [AE, D]

epochs = 75

trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, lambd=100)

#trains the model
trainer.train()


state_dict = torch.load('./model/pix2pix_G.pth')
AE.load_state_dict(state_dict)

model_list = [AE]

tester = Tester(test_loader, model_list)
tester.test()
