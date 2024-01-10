import math
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import autograd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# full connected forward and backward of FA
class FA_Function(autograd.Function):
    # same as reference linear function, but with additional fa tensor for backward
    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # used GPU
            weight_fa = weight_fa.to(device)
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
            if grad_bias.dim() == 0:
                grad_bias = grad_bias.view(1)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


# full connected layer of FA
class FA_linear(nn.Module):
    def __init__(self, input_features, output_features, FA_rule):
        super(FA_linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.FA_rule = FA_rule

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        # B value
        self.weight_fa = Variable(torch.FloatTensor(output_features, input_features), requires_grad=False)

        # initialized weight
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.bias, 0)

        if self.FA_rule == 'FA':
            torch.nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        elif self.FA_rule == 'FA_Ex-100%':
            torch.nn.init.constant_(self.weight_fa, 0.5)
        elif self.FA_rule == 'FA_Ex-0%':
            torch.nn.init.constant_(self.weight_fa, -0.5)
        elif self.FA_rule == 'FA_Ex-80%' or 'FA_Ex-50%' or 'FA_Ex-20%':
            torch.nn.init.uniform_(self.weight_fa)

    def forward(self, input):
        return FA_Function.apply(input, self.weight, self.weight_fa, self.bias)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# convolutional forward and backward of FA
class FA_conv_Function(autograd.Function):
    # same as reference linear function, but with additional fa tensor for backward

    @staticmethod
    def forward(context, input, weight, weight_fa, out_channels, image_size, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = torch.unsqueeze(input, dim=1)
        output = output.repeat(1, out_channels, 1, 1)
        output = torch.mul(weight, output)

        if bias is not None:
            output = output.sum(2) + bias
            output = output.view(input.size()[0], out_channels, image_size, image_size)

        return output

    @staticmethod
    def backward(context, grad_output):
        # input, weight, weight_backward, bias, bias_backward = context.saved_tensors
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_out_channels = grad_image_size = grad_bias = None

        if context.needs_input_grad[0]:
            # used GPU
            weight_fa = weight_fa.to(device)

            grad_output = grad_output.view(grad_output.size()[0], grad_output.size()[1], -1)
            grad_output = grad_output.unsqueeze(dim=2)
            grad_input = grad_output.mul(weight_fa)
            grad_input = grad_input.sum(1)

        if context.needs_input_grad[1]:
            grad_output = grad_output.view(grad_output.size()[0], grad_output.size()[1], -1)
            input = input.unsqueeze(dim=1)
            grad_output = grad_output.unsqueeze(dim=2)
            grad_weight = grad_output.mul(input)

        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
            if grad_bias.dim() == 0:
                grad_bias = grad_bias.view(1)

        return grad_input, grad_weight, grad_weight_fa, grad_out_channels, grad_image_size, grad_bias


# Convolutional layer of FA
class FA_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, FA_rule):
        super(FA_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.FA_rule = FA_rule

        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, (self.kernel_size ** 2) * self.in_channels, self.image_size ** 2))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels, self.image_size ** 2))

        # B value
        self.weight_fa = Variable(
            torch.FloatTensor(self.out_channels, (self.kernel_size ** 2) * self.in_channels, self.image_size ** 2),
            requires_grad=False)

        # initialized weight
        torch.nn.init.uniform_(self.weight, a=-1 / kernel_size, b=1 / kernel_size)
        torch.nn.init.constant_(self.bias, 0)
        if self.FA_rule == 'FA':
            torch.nn.init.uniform_(self.weight_fa, a=-1 / kernel_size, b=1 / kernel_size)
        elif self.FA_rule == 'FA_Ex-100%':
            torch.nn.init.constant_(self.weight_fa, 0.5)

    def forward(self, input):
        return FA_conv_Function.apply(input, self.weight, self.weight_fa, self.out_channels, self.image_size, self.bias)

