import torch
from torch import nn

def get_outputpadding(input_size, output_size, kernel_size, stride, padding):
    outputpadding = output_size - (input_size - 1) * stride + 2 * padding - kernel_size
    return outputpadding

input = torch.Tensor(2, 3, 7, 7)
conv = nn.Conv2d(3, 1, 3, 6, 1)
output1 = conv(input)
print(output1.shape)
conv_transpose = nn.ConvTranspose2d(1, 3, 2, 6, 1, get_outputpadding(2, 7, 2, 6, 1))
output2 = conv_transpose(output1)
print(output2.shape)


