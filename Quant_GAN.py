import torch
import torch.nn as nn

'''
Temporal Block
'''
class temporal_block(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, kernel_size, dilation,
                 stride=1, padding='same', bias=True):
        super(temporal_block, self).__init__()
        '''
        Define convolution layers
        '''
        self.convolution_1 = nn.Conv1d(num_in, num_hidden, kernel_size, stride, padding, dilation, bias=bias)
        self.convolution_2 = nn.Conv1d(num_hidden, num_out, kernel_size, stride, padding, dilation, bias=bias)

        '''
        Define the temporal block
        '''
        self.temp_net = nn.Sequential(
            self.convolution_1,
            nn.PReLU(),
            self.convolution_2,
            nn.PReLU()

        )

        # define downsampling
        if num_in != num_out:
            self.downsampling = nn.Conv1d(num_in, num_out, 1)
        else:
            self.downsampling = None

        # initializing weights
        self.initialize_weights()

    def initialize_weights(self):
        # initializing weights
        self.convolution_1.weight.data.normal_(std=0.01)
        self.convolution_2.weight.data.normal_(std=0.01)
        if self.downsampling != None:
            self.downsampling.weight.data.normal_(std=0.01)

    # define forward path
    def forward(self, x):
        output = self.temp_net(x)
        if self.downsampling != None:
            x = self.downsampling(x)

        return output + x


'''
Temporal Convolutional Networks (TCNs)
'''
class TCN(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(TCN, self).__init__()

        '''
        Define the layers
        '''
        layers = [temporal_block(num_in if i == 0 else num_hidden,
                                 num_hidden, num_hidden, 2 if i > 0 else 1,
                                 1 if i == 0 else 2 ** i)
                  for i in range(7)]

        self.temp_net = nn.Sequential(*layers)
        self.convolution_1 = nn.Conv1d(num_hidden, num_out, 1)

        self.initialize_weights()

    def initialize_weights(self):
        # initializing weights
        self.convolution_1.weight.data.normal_(std=0.01)

    # defining
    def forward(self, x):
        return self.convolution_1(self.temp_net(
            torch.transpose(x, 1, 2))).transpose(1, 2)


'''
Define generator
'''
class generator(nn.Module):
    def __init__(self, num_in, num_out):
        super(generator, self).__init__()
        self.temp_net = TCN(num_in, 80, num_out)

    def forward(self, x):
        return torch.tanh(self.temp_net(x))

'''
Define discriminator
'''
class discriminator(nn.Module):
    def __init__(self, num_in, num_out):
        super(discriminator, self).__init__()
        self.temp_net = TCN(num_in, 80, num_out)

    def forward(self, x):
        return torch.sigmoid(self.temp_net(x))
