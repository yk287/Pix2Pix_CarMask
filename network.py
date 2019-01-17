

#input is 224

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def if_tensor(tensor):

    if isinstance(tensor, torch.Tensor):
        return True

class conv_down(nn.Module):

    def __init__(self, channel_input, channel_output, kernel=3, stride=1, padding=0, Norm=True, Dropout=0.2):
        super(conv_down, self).__init__()

        steps = [nn.Conv2d(channel_input, channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.BatchNorm2d(channel_output))
        steps.append(nn.LeakyReLU(0.2))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class conv_up(nn.Module):

    def __init__(self, channel_input, channel_output, kernel=3, stride=1, padding=0, Norm=True, Dropout=0):
        super(conv_up, self).__init__()

        steps = [nn.ConvTranspose2d(channel_input, channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.BatchNorm2d(channel_output))
        steps.append(nn.ReLU(inplace=True))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x, skip_input=None):

        if if_tensor(skip_input):
            x = torch.cat((x, skip_input), 1)

        return self.model(x)

class AutoEncoder_Unet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(AutoEncoder_Unet, self).__init__()

        self.U_down1 = conv_down(channel_input=in_channel, channel_output=16, kernel=4, stride=2, padding=1, Norm=False)
        self.U_down2 = conv_down(channel_input=16, channel_output=32, kernel=4, stride=2, padding=1)
        self.U_down3 = conv_down(channel_input=32, channel_output=64, kernel=4, stride=2, padding=1)
        self.U_down4 = conv_down(channel_input=64, channel_output=128, kernel=4, stride=2, padding=1)
        self.U_down5 = conv_down(channel_input=128, channel_output=256, kernel=4, stride=2, padding=1)
        self.U_down6 = conv_down(channel_input=256, channel_output=512, kernel=4, stride=1, padding=1)
        self.U_down7 = conv_down(channel_input=512, channel_output=512, kernel=4, stride=1, padding=1)
        self.U_down8 = conv_down(channel_input=512, channel_output=1024, kernel=4, stride=1, padding=1)
        self.U_down9 = conv_down(channel_input=1024, channel_output=1024, kernel=4, stride=1, padding=1,  Norm=False)

        self.U_up1 = conv_up(channel_input=1024, channel_output=1024, kernel=2, stride=1)
        self.U_up2 = conv_up(channel_input=2048, channel_output=512, kernel=2, stride=1)
        self.U_up3 = conv_up(channel_input=1024, channel_output=512, kernel=2, stride=1)
        self.U_up4 = conv_up(channel_input=1024, channel_output=256, kernel=2, stride=1)
        self.U_up5 = conv_up(channel_input=512, channel_output=128, kernel=2, stride=2)
        self.U_up6 = conv_up(channel_input=256, channel_output=64, kernel=2, stride=2)
        self.U_up7 = conv_up(channel_input=128, channel_output=32, kernel=2, stride=2)
        self.U_up8 = conv_up(channel_input=64, channel_output=16, kernel=2, stride=2)
        self.U_up9 = conv_up(channel_input=32, channel_output=out_channel, kernel=2, stride=2)

        self.tan_output = nn.Sequential(
            nn.Tanh()
        )
    def forward(self, x):

        input_down1 = self.U_down1(x)
        input_down2 = self.U_down2(input_down1)
        input_down3 = self.U_down3(input_down2)
        input_down4 = self.U_down4(input_down3)
        input_down5 = self.U_down5(input_down4)
        input_down6 = self.U_down6(input_down5)
        input_down7 = self.U_down7(input_down6)
        input_down8 = self.U_down8(input_down7)
        input_down9 = self.U_down9(input_down8)

        x = self.U_up1(input_down9)
        x = self.U_up2(x, input_down8)
        x = self.U_up3(x, input_down7)
        x = self.U_up4(x, input_down6)
        x = self.U_up5(x, input_down5)
        x = self.U_up6(x, input_down4)
        x = self.U_up7(x, input_down3)
        x = self.U_up8(x, input_down2)
        x = self.U_up9(x, input_down1)

        return self.tan_output(x)

class discriminator(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(discriminator, self).__init__()

        self.U_down1 = conv_down(channel_input=in_channel, channel_output=16, kernel=2, stride=2, Norm=False)
        self.U_down2 = conv_down(channel_input=16, channel_output=32, kernel=2, stride=2)
        self.U_down3 = conv_down(channel_input=32, channel_output=32, kernel=2, stride=2)
        self.U_down4 = conv_down(channel_input=32, channel_output=64, kernel=2, stride=2)
        self.U_down5 = conv_down(channel_input=64, channel_output=128, kernel=2, stride=2, padding=1)
        self.U_down6 = conv_down(channel_input=128, channel_output=256, kernel=2, stride=2)
        self.U_down7 = conv_down(channel_input=256, channel_output=256, kernel=2, stride=2, Norm=False)

        self.linear = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256 * 2 * 2),
            #nn.BatchNorm1d(256 * 2 * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(256 * 2 * 2, 128 * 2 * 2),
            #nn.BatchNorm1d(128 * 2 * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(128 * 2 * 2, 64 * 2 * 2),
            #nn.BatchNorm1d(64 * 2 * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * 2 * 2, 32 * 2 * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(32 * 2 * 2, 16 * 2 * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(16 * 2 * 2, out_channel)
        )
    def forward(self, x):

        x = self.U_down1(x)
        x = self.U_down2(x)
        x = self.U_down3(x)
        x = self.U_down4(x)
        x = self.U_down5(x)
        x = self.U_down6(x)
        x = self.U_down7(x)

        x = self.linear(x.view(x.shape[0], -1))

        return x


