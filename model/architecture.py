import torch.nn as nn
from .import block as B
import torch
from thop import profile
from .channel import Channel


class PRAF_JSCC(nn.Module):
    def __init__(self, in_nc=3, nf=256, num_modules=6, out_nc=3, upscale=2, c=8, channel_type='awgn'):
        super(PRAF_JSCC, self).__init__()
        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, kernel_size=3, stride=2),
                                      nn.LeakyReLU(0.05),
                                      B.conv_layer(nf, nf, kernel_size=3, stride=2))

        self.PRAF1 = B.PRAF(in_channels=nf)
        self.PRAF2 = B.PRAF(in_channels=nf)
        self.PRAF3 = B.PRAF(in_channels=nf)
        self.PRAF4 = B.PRAF(in_channels=nf)
        self.PRAF5 = B.PRAF(in_channels=nf)
        self.PRAF6 = B.PRAF(in_channels=nf)

        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.Sent = nn.Conv2d(nf, c, kernel_size=3, stride=1)
        self.SAD = B.CCA(nf)
        self.Receive = nn.ConvTranspose2d(c, nf, kernel_size=3, stride=1)
        self.Restruct = nn.Sequential(nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.LeakyReLU(),
                                      nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1, output_padding=1))
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.channel = Channel(channel_type= channel_type)
        self.Sigmoid = nn.Sigmoid()
    def forward(self, input, SNR):
        out_fea = self.fea_conv(input)
        out_B1 = self.PRAF1(out_fea, SNR)
        out_B2 = self.PRAF2(out_B1, SNR)
        out_B3 = self.PRAF3(out_B2, SNR)
        out_B4 = self.PRAF4(out_B3, SNR)
        out_B5 = self.PRAF5(out_B4, SNR)
        out_B6 = self.PRAF6(out_B5, SNR)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        Sent = self.Sent(out_lr)
        noise_lr = self.channel(Sent, snr_db=SNR)
        Receiver = self.Receive(noise_lr)
        receive = self.SAD(Receiver, SNR)
        receive = self.Restruct(receive)
        output = self.upsampler(receive)
        output = self.Sigmoid(output)
        return output


if __name__ == '__main__':
    upscale = 2
    window_size = 8
    height = 2048 // upscale
    width = 1024 // upscale
    model = PRAF_JSCC(nf=256, c=8)
    model.cuda()
    # print(model)
    x = torch.randn((1, 3, height, width)).cuda()
    snr_tensor = torch.tensor([1.0], dtype=torch.float32).cuda()
    flops, params = profile(model, inputs=(x, snr_tensor), verbose=False)
    print(flops / 1e9)
    print(params / 1e6)
