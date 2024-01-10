import time

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.autograd import Variable

import random
import matplotlib.pyplot as plt

import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
plt.rcParams["font.size"] = 18
fig = plt.figure()

# seed
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# downloaded dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

#"""
# mnist
train_dataset = datasets.MNIST(
    './data_mnist',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    './data_mnist',
    train=False,
    download=True,
    transform=transform
)
#"""

"""
# cifar10
train_dataset = datasets.CIFAR10(
    './data_cifar10',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.CIFAR10(
    './data_cifar10',
    train=False,
    download=True,
    transform=transform
)
"""

if __name__ == '__main__':
    # measured time
    start = time.time()

    # hyper parameter
    batch_size = 32
    epochs = 10
    lr = 0.01

    # data processing
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    # loss log
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    true_test_acc_history = []

    # defined model, loss_function, optimizer
    Neuralnet = model.fa_lenet()
    Neuralnet = Neuralnet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.SGD(Neuralnet.parameters(), lr=lr)

    print("----------------------------------------------------------------")
    print(
        f"batch_size: {batch_size}, lr: {lr}, epoch: {epochs}")

    for epoch in range(epochs):
        train_loss_sum = 0
        train_correct = 0

        # train mode----------------------------------------
        Neuralnet.train()

        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs), Variable(labels)

            # GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # initialized optimizer
            optimizer.zero_grad()

            # predicted
            outputs = Neuralnet(inputs)

            # calculated loss
            loss = criterion(outputs, labels)
            train_loss_sum += loss

            # calculated accuracy rate
            train_pred = outputs.argmax(1)
            train_correct += train_pred.eq(labels.view_as(train_pred)).sum().item()

            # updated weight
            loss.backward()
            optimizer.step()

            # initialized optimizer
            optimizer.zero_grad()

        # appended loss to logs
        train_loss_history.append(train_loss_sum.item() / len(train_loader))
        train_acc_history.append(100 * train_correct / len(train_dataset))

        print(f"Epoch: {epoch + 1}/{epochs}, Train_Loss: {train_loss_sum.item() / len(train_loader)},"
              f"Accuracy: {100 * train_correct / len(train_dataset)}% ({train_correct}/{len(train_dataset)})")

        # value mode------------------------------------------------------------------------------------------
        test_loss_sum = 0
        correct = 0
        Neuralnet.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                # GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # predicted
                outputs = Neuralnet(inputs)

                # calculated loss
                loss = criterion(outputs, labels)
                test_loss_sum += loss

                # calculated accuracy rate
                pred = outputs.argmax(1)
                correct += pred.eq(labels.view_as(pred)).sum().item()

            # appended loss to logs
            test_loss_history.append(test_loss_sum.item() / len(test_loader))
            test_acc_history.append(100 * correct / len(test_dataset))

            print(f"Epoch: {epoch + 1}/{epochs}, Test_Loss: {test_loss_sum.item() / len(test_loader)}, "
                  f"Accuracy: {100 * correct / len(test_dataset)}% ({correct}/{len(test_dataset)})")

    # measured time
    elapsed_time = time.time() - start
    print("elapsed time:{0}".format(elapsed_time) + "[sec]")
    plt.plot(list(range(1, epochs + 1)), train_loss_history, label="train_loss")
    plt.plot(list(range(1, epochs + 1)), test_loss_history, label="test_loss")

plt.subplots_adjust(left=0.18, bottom=0.13, right=0.99, top=0.97)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.ylim(0.5, 2)
plt.legend()
fig.savefig("figure.svg")
plt.show()
