import torch
import torch.nn as nn
import torch.nn.functional as F

import FA_function

"""
cifar10
"""
class fa_cnn_net(nn.Module):
    def __init__(self):
        super(fa_cnn_net, self).__init__()
        # activation function
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.pool = nn.AvgPool2d(3, stride=2)

        # convolution network
        self.unfold1 = nn.Unfold(kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv1 = FA_function.FA_Conv2d(3, 32, 5, 32, 'FA')
        self.conv2 = FA_function.FA_Conv2d(32, 64, 5, 15, 'FA')
        self.conv3 = FA_function.FA_Conv2d(64, 64, 5, 7, 'FA')

        self.fa_fc1 = FA_function.FA_linear(64 * 3 * 3, 128, 'FA')
        self.fa_fc2 = FA_function.FA_linear(128, 10, 'FA')

    def forward(self, x):
        x = self.unfold1(x)
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.unfold1(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.unfold1(x)
        x = self.conv3(x)
        x = self.tanh(x)
        x = self.pool(x)

        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fa_fc1(x)
        x = self.tanh(x)
        x = self.fa_fc2(x)

        return F.log_softmax(x, dim=1)


class conv2d_net(nn.Module):
    def __init__(self):
        super(conv2d_net, self).__init__()
  
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.pool = nn.AvgPool2d(3, stride=2)

        self.conv1 = nn.Conv2d(3, 32, 5, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, 5, padding=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, 5, padding=(2, 2))

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class fa_fc_net(nn.Module):
    def __init__(self):
        super(fa_fc_net, self).__init__()

        self.relu = nn.ReLU()
        self.fa_fc1 = FA_function.FA_linear(3 * 32 * 32, 128, 'FA')
        self.fa_fc2 = FA_function.FA_linear(128, 10, 'FA')

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fa_fc1(x)
        x = self.relu(x)
        x = self.fa_fc2(x)

        return F.log_softmax(x, dim=1)
    

class fc_net(nn.Module):
    def __init__(self):
        super(fc_net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    
    
"""
MNIST
"""
class fa_lenet(nn.Module):
    def __init__(self):
        super(fa_lenet, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.pool = nn.MaxPool2d(2, stride=2)

        # convolution network
        self.unfold1 = nn.Unfold(kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv1 = FA_function.FA_Conv2d(1, 6, 5, 28, 'FA')
        self.conv2 = FA_function.FA_Conv2d(6, 16, 5, 14, 'FA')

        self.fa_fc1 = FA_function.FA_linear(16 * 7 * 7, 128, 'FA')
        self.fa_fc2 = FA_function.FA_linear(128, 10, 'FA')

    def forward(self, x):
        x = self.unfold1(x)
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.unfold1(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fa_fc1(x)
        x = self.tanh(x)
        x = self.fa_fc2(x)

        return F.log_softmax(x, dim=1)
    
    
class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1, 6, 5, padding=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5, padding=(2, 2))

        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
