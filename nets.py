import torch
from torch import nn
from utils import get_outputpadding


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            ConvolutionalLayer(1, 128, 3, 2, 1),  # n,128,14,14
            ConvolutionalLayer(128, 256, 3, 2, 1),  # n,256,7,7
            ConvolutionalLayer(256, 512, 3, 2),  # n,512,3,3
            ConvolutionalLayer(512, 2, 3, 1)  # n,2,1,1
        )

    def forward(self, data):
        conv_layer = self.conv_layer(data)
        miu, log_sigma = conv_layer.chunk(2, dim=1)
        return miu, log_sigma


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_layer = nn.Sequential(
            ConvTransposeLayer(128, 512, 3, 1, get_outputpadding(1, 3, 3, 1, 0)),  # n,128,3,3
            ConvTransposeLayer(512, 256, 3, 2, get_outputpadding(3, 7, 3, 2, 0)),  # n,256,7,7
            ConvTransposeLayer(256, 128, 3, 2, 1, get_outputpadding(7, 14, 3, 2, 1)),  # n,128,14,14
            nn.ConvTranspose2d(128, 1, 3, 2, 1, get_outputpadding(14, 28, 3, 2, 1)),  # n,1,28,28
            nn.Tanh()
        )

    def forward(self, data):
        return self.conv_transpose_layer(data)


class Total_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data, distribution):
        miu, log_sigma = self.encoder(data)
        data = distribution * torch.exp(log_sigma) + miu
        data = data.permute(0, 3, 1, 2)
        output = self.decoder(data)
        return miu, log_sigma, output


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


if __name__ == '__main__':
    net = Total_Net()
    input = torch.Tensor(3, 1, 28, 28)
    distribution = torch.randn(128)
    miu, log_sigma, output = net(input, distribution)
    print(output.shape)
    encoder_p = sum(p.numel() for p in net.encoder.parameters())
    decoder_p = sum(p.numel() for p in net.decoder.parameters())
    print(encoder_p)
    print(decoder_p)
