import torch
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
  # k: convolution kernel_size, 
  # c: convolution number of kernels (feature maps)
  # p: padding size
  # pk: pooling kernerl_size
  def __init__(self, k1, c1, p1, pk1, k2, c2, p2, pk2):
    #IMPORTANT number of channel is not size of image
    super().__init__()
    # in_channel, out_channel, kernel_size (size of the filter) 5x5 filter
    # taking in 3 channels and having 6 filters
    self.conv1 = nn.Conv2d(3, c1, k1)
    # kernel_size, stride
    self.pool = nn.MaxPool2d(pk1, 2)
    self.conv2 = nn.Conv2d(c1, c2, k2)
    # flatten the convolutional layer into a string
    self.fc1 = nn.Linear(c2 * k2 * k2, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def train(self, train_loader, epoch_num, criterion, optimizer):
    #train mode
    self.train()
    for epoch in range(epoch_num):
      for i, data in enumerate(train_loader):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()


#running a cnn:
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(), 
     transforms.Normalize((0.1307,), (0.3081,)),])


batch_size = 4

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

#returns a DataLoader which can be iterated over with for each loop
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

