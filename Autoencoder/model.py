import math
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import autograd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# forward and feedback of feedback alignment
class FA_Function(autograd.Function):
    
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


class FAModule(nn.Module):
    def __init__(self, input_features, output_features, FA_rule):
        super(FAModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.FA_rule = FA_rule

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        # FA_B value
        self.weight_fa = Variable(torch.FloatTensor(output_features, input_features), requires_grad=False)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.FA_rule == 'FA':
            torch.nn.init.kaiming_uniform_(self.weight_fa, a=0.5)
        elif self.FA_rule == 'FA_Ex-100%':
            torch.nn.init.constant_(self.weight_fa, 0.5)
        elif self.FA_rule == 'FA_Ex-0%':
            torch.nn.init.constant_(self.weight_fa, -0.5)
        elif self.FA_rule == 'FA_Ex-80%' or 'FA_Ex-50%' or 'FA_Ex-20%':
            torch.nn.init.uniform_(self.weight_fa)

        torch.nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return FA_Function.apply(input, self.weight, self.weight_fa, self.bias)


# autoencoder model
class FA_Autoencoder(nn.Module):
    def __init__(self, learning_rule, in_features, middle_neuron, out_features):
        super(FA_Autoencoder, self).__init__()
        self.in_features = in_features
        self.middle_neuron = middle_neuron
        self.out_features = out_features
        self.learning_rule = learning_rule
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

        if self.learning_rule == 'BP':
            self.linear1 = nn.Linear(self.in_features, self.middle_neuron)
            self.linear2 = nn.Linear(self.middle_neuron, self.out_features)

        elif self.learning_rule == 'FA':
            self.linear1 = FAModule(self.in_features, self.middle_neuron, self.learning_rule)
            self.linear2 = FAModule(self.middle_neuron, self.out_features, self.learning_rule)

        elif self.learning_rule == 'FA_Ex-100%' or 'FA_Ex-80%' or 'FA_Ex-50%' or 'FA_Ex-20%' or 'FA_Ex-0%':
            self.linear1 = FAModule(self.in_features, self.middle_neuron, self.learning_rule)
            self.linear2 = FAModule(self.middle_neuron, self.out_features, self.learning_rule)

    def forward(self, inputs):
        # encorder
        x = self.linear1(inputs)
        x = self.relu1(x)
        # decorder
        x = self.linear2(x)
        y = self.sigmoid1(x)

        return y
