import torch
import torch.nn.functional as F


def conv2d(x, W):
    '''1 dimentional convolution defined in the paper
    the function's name is not appropriate and
    we didn't change that'''
    return torch.nn.Conv2d(x, W, strides=[1, 100, 1, 1], padding='SAME')


class MyCNN(torch.nn.Module):
    """
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    """

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self, params):

        super(MyCNN, self).__init__()

        self.params = params

        # Architecture
        # To understand what will be the output size, see :
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=3, stride=1,
                                     padding=1)
        self.conv2 = torch.nn.Conv2d(5, 7, kernel_size=3, stride=1,
                                     padding=1)
        self.conv3 = torch.nn.Conv2d(7, 10, kernel_size=3, stride=1,
                                     padding=1)
        self.conv4 = torch.nn.Conv2d(10, 7, kernel_size=3, stride=1,
                                     padding=1)
        self.conv5 = torch.nn.Conv2d(7, 5, kernel_size=3, stride=1,
                                     padding=1)
        self.conv6 = torch.nn.Conv2d(5, 1, kernel_size=3, stride=1,
                                     padding=1)
        
        self.double()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        return x