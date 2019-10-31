import torch
from torch import nn
import PIL.Image as pimg


def get_outputpadding(input_size, output_size, kernel_size, stride, padding):
    outputpadding = output_size - (input_size - 1) * stride + 2 * padding - kernel_size
    return outputpadding


input = torch.Tensor(2, 3, 28, 28)
conv = nn.Conv2d(3, 1, 3, 2, 1)
output1 = conv(input)
print(output1.shape)
conv_transpose = nn.ConvTranspose2d(1, 3, 3, 2, 1, get_outputpadding(14, 28, 3, 2, 1))
output2 = conv_transpose(output1).permute(0, 3, 2, 1)
print(output2.shape)
output2 = output2.detach().numpy()[0]
image = pimg.fromarray(output2, "RGB")
print(image)
image.show()
