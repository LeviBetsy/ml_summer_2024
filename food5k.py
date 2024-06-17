#dataset location wrongs

import torch
import os
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAccuracy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torchvision.io import read_image
from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights

# device = 'metals' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))
# device_name = "cpu"
device_name = "mps"
device = torch.device(device_name)

alexnet_model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)

vgg16_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

#freezing
alexnet_model.eval()
for param in alexnet_model.parameters():
  param.requires_grad = False

vgg16_model.eval()
for param in vgg16_model.parameters():
  param.requires_grad = False

class Food5kDataset(torch.utils.data.Dataset):
  def __init__(self, split="train", transform=None):
    if (split == "train"):
      self.img_dir = "data/food5k/training"
    elif (split == "test"):
      self.img_dir = "data/food5k/evaluation"

    #food and non food directory
    self.food_dir = os.path.join(self.img_dir, "food")
    self.len_food_dir = len([name for name in os.listdir(self.food_dir)])
    self.non_food_dir = os.path.join(self.img_dir, "non_food")
    self.len_non_food_dir = len([name for name in os.listdir(self.non_food_dir)])
    #transform
    self.transform = transform

  def __len__(self):
    return self.len_food_dir + self.len_non_food_dir

  def __getitem__(self, idx):
    if idx < self.len_food_dir:
      img_path = os.path.join(self.food_dir, f"{idx}.jpg")
      label = 1
    else:
      img_path = os.path.join(self.non_food_dir, f"{idx % self.len_food_dir}.jpg")
      label = 0
    # print(img_path)
    image = read_image(img_path, torchvision.io.ImageReadMode.RGB)
    # print(image[0][0])
    if self.transform:
      image = self.transform(image)
    return image, label

dataset_ratio = 0.05
transforms = AlexNet_Weights.IMAGENET1K_V1.transforms()
# print(transforms) TODO: ask tuba about transform
batch_size = 4

#TODO: ask tuba about validation

#training data
#trainset has 3000 images
trainset = Food5kDataset("train", transforms)
trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

#test data
#testset has 1000 images
testset = Food5kDataset("test", transforms)
testset, _ = torch.utils.data.random_split(testset, [dataset_ratio, 1 - dataset_ratio])
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
# TODO: ask Tuba why len testset 8, batchsize 4 but tesloader returns length of 6??

print(f"length of training set: {len(trainset)}")
print(f"length of testing set: {len(testset)}")

# #show some image
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 2, 5
# # indx = [0,1500, 1, 1501, 2, 1502, 3, 1503, 4, 1504]
# for i in range(cols * rows):
#     img, label = trainset[i]
#     img = img.permute(1, 2, 0)
#     figure.add_subplot(rows, cols, i + 1)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_iter = iter(trainloader)
dummy_input, lab = next(train_iter)
# print(dummy_input.shape)
dummy_input = dummy_input.to(device)

class AlexNetFc6(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.features = nn.Sequential(
        alexnet_model.features,
        alexnet_model.avgpool
    )
    self.fc6 = alexnet_model.classifier[:2] #slice the module to only stop until fc 6
    print(self.fc6)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.fc6(x)
    return x

class VGG16Fc6(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.features = vgg16_model.features
    self.avgpool = vgg16_model.avgpool
    self.fc6 = vgg16_model.classifier[:1] #slice the module to only stop until fc 6
    print(self.fc6)

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc6(x)
    return x

alexnetfc6 = AlexNetFc6().to(device)
alexnetfc6.eval()
vgg16fc6 = VGG16Fc6().to(device)
vgg16fc6.eval()

class ConcatFeatureNet(nn.Module):
  def __init__(self, pretrained_models, dummy_input): #hp stands for hyperparameters
    super().__init__()
    self.pretrained_models = pretrained_models

    ft_concat_size = 0

    with torch.no_grad():
      for model in pretrained_models:
        dummy_output = model(dummy_input)
        ft_concat_size += dummy_output.size(1) #ignore first number which is batch size

    print(f"Number of features: {ft_concat_size}")

    self.layers = nn.Sequential(
      nn.ReLU(), #TODO: ask Dr.Tuba should I relu all the features here, because technically they went through a set of weights
      nn.Linear(ft_concat_size, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),

      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(256, 2),
      nn.Softmax(dim = 1) #apply soft max to the second dimension, ignoring batch
    )

  def forward(self, x):
    ft_concat = torch.empty(0).to(device)
    for pretrained_model in self.pretrained_models:
      with torch.no_grad():
        ft = pretrained_model(x)
        ft = ft.to(device)
        # print(ft.shape)
        #concatenate the features, keeping them the same batch dimension, only concatenating the content (dim = 1)
        ft_concat = torch.cat((ft_concat, ft), dim = 1).to(device)


    ft_concat = ft_concat.to(device)
    x = self.layers(ft_concat)
    return x

  def train_model(self, trainloader, epoch_num, criterion, optimizer):
    for epoch in range(epoch_num):
      self.train()
      print(str(epoch + 1) + "th epoch")
      for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        print(inputs.shape)
        print(labels.shape)
        print(inputs)
        print(labels)


        optimizer.zero_grad()
        outputs = self(inputs)


        # print(outputs)
        # print(labels)

        #outputs.shape is [batch_size, number_of_classes] so like [[0.1, 0.9],[0.4, 0.6]] for 2 classes and 2 batch
        #labels.shape is [batch_size] so like [1,0]
        loss = criterion(outputs, labels)
        # print(loss)

        loss.backward()

        optimizer.step()
      print(f"Accuracy at {epoch + 1}th epoch: {self.assess_accuracy(testloader)}")
  def assess_accuracy(self, testloader):
    self.eval()
    predictions = torch.empty(0).to(device)
    actual_labels = torch.empty(0).to(device)
    with torch.no_grad():
      for data in testloader:
        images, labels = data
        labels = labels.to(device)
        # print(labels.shape)
        actual_labels = torch.cat((actual_labels, labels), dim = 0)

        images = images.to(device)
        outputs = self(images)
        predictions = torch.cat((predictions, outputs), dim = 0)

      # print(predictions.shape)
      # print(actual_labels.shape)

      metric = MulticlassAccuracy()
      metric.update(predictions, actual_labels)
      return metric.compute()

# with torch.no_grad():
#   vgg16fc6(dummy_input)

concat = ConcatFeatureNet([alexnetfc6, vgg16fc6], dummy_input).to(device)

criterion = nn.CrossEntropyLoss()
epoch_num = 1
learning_rate = 0.001

optimizer = optim.Adam(concat.parameters(), lr=learning_rate)
concat.train_model(trainloader, epoch_num, criterion, optimizer)