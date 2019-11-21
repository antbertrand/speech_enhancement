import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, batch_size, NEFF, N_IN, N_OUT, DECAY=0.999)::
        '''NEFF: number of effective FFT points
        N_IN: number of input frames into the nets
        N_OUT: only tested for 1, errors may occur for other number
        DECAY: decay for global mean and var estimation using batch norm
        '''
        
        super(CNN, self).__init__()
        
        self.batch_size = batch_size
        self.NEFF = NEFF
        self.N_IN = N_IN
        self.N_OUT = N_OUT
        self.DECAY = DECAY
        
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.fc1(x)
        return x