import torch
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights

#can also use pretrained to only do feature extraction and not image classification

alexnet_model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)

vgg16_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

#freezing
for param in alexnet_model.parameters():
  param.requires_grad = False

for param in vgg16_model.parameters():
  param.requires_grad = False

#Alexnet and VGG16 has the same transformation so we can use the same input dataset

batch_size = 4

dataset_ratio = 0.001
transforms = AlexNet_Weights.IMAGENET1K_V1.transforms()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
device = torch.device(device)


# train dataset

trainset = torchvision.datasets.Food101(root='./data', split="train",
                                        download=True, transform=transforms)

trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
#test dataset

testset = torchvision.datasets.Food101(root='./data', split="test",
                                       download=True, transform=transforms)

testset, _ = torch.utils.data.random_split(testset, [dataset_ratio, 1 - dataset_ratio])

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transforms)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transforms)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

print(f"length of training set: {len(trainset)}")
print(f"length of testing set: {len(testset)}")

dummy_input, _ = next(iter(trainloader))
dummy_input = dummy_input.to(device)

class AlexNetFc6(nn.Module):
  def __init__(self, dummy_input): #hp stands for hyperparameters
    super().__init__()
    self.features = nn.Sequential(
        alexnet_model.features,
        alexnet_model.avgpool
    )
    self.fc6 = alexnet_model.classifier[:4] #slice the module to only stop until fc 6
    # print(self.fc6)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.fc6(x)
    return x

class VGG16Fc6(nn.Module):
  def __init__(self, dummy_input): #hp stands for hyperparameters
    super().__init__()
    self.features = vgg16_model.features
    self.avgpool = vgg16_model.avgpool
    self.fc6 = vgg16_model.classifier[:3] #slice the module to only stop until fc 6
    print(self.fc6)

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc6(x)
    return x

class ConcatFeatureNet(nn.Module):
  def __init__(self, pretrained_models, dummy_input): #hp stands for hyperparameters
    super().__init__()
    self.pretrained_models = pretrained_models

    ft_concat_size = 0

    for model in pretrained_models:
      # model = model.to(device)
      dummy_output = model(dummy_input.to(device))
      ft_concat_size += dummy_output.size(1) #ignore first number, batch size

    print(f"Number of features: {ft_concat_size}")

    self.fc = nn.Linear(ft_concat_size, 101)

  def forward(self, x):
    ft_concat = torch.empty(0).to(device)
    for model in self.pretrained_models:
      model = model.to(device)
      ft = model(x)
      ft_concat = torch.cat((ft_concat, ft), dim = 1).to(device)


    ft_concat = ft_concat.to(device)
    x = self.fc(ft_concat)

    return x

  def train_model(self, trainloader, epoch_num, criterion, optimizer):
    for epoch in range(epoch_num):
      self.train()
      print(str(epoch + 1) + "th epoch")
      for i, data in enumerate(trainloader):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()
          # forward + backward + optimize
          outputs = self(inputs)

          #compare how output and actual label is different
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()
      print(self.assess_accuracy(testloader))


  def assess_accuracy(self, testloader):
    self.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = self(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return (100 * correct // total)

criterion = nn.CrossEntropyLoss()
epoch_num = 1
learning_rate = 0.001


# fc6_alex = torch.nn.Sequential(*list(alexnet_model.classifier.children())[:7])
# fc6_vgg = torch.nn.Sequential(*list(vgg16_model.classifier.children())[:7])
concat_model = ConcatFeatureNet([AlexNetFc6(dummy_input).to(device), VGG16Fc6(dummy_input).to(device)], dummy_input)
concat_model = concat_model.to(device)

optimizer = optim.Adam(concat_model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

concat_model.train_model(trainloader, epoch_num, criterion, optimizer)