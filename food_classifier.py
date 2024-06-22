import pandas as pd
import torch
import numpy as np
from torchvision.models import AlexNet_Weights
import os
import torch.nn as nn
import time
from torchvision.io import read_image
import torch.optim as optim
import torchvision
from torcheval.metrics import MulticlassAccuracy
import frozen_features

device_name = "mps"
device = torch.device(device_name)


#                                         NN
class Food5kClassifier(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.layers = nn.Sequential(
      nn.ReLU(),
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
    x = self.layers(x)
    return x

class Food101kClassifier(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.layers = nn.Sequential(
      nn.ReLU(),
      nn.Linear(ft_concat_size, 4000),
      nn.BatchNorm1d(4000),
      nn.ReLU(),

      nn.Linear(4000, 1000),
      nn.BatchNorm1d(1000),
      nn.ReLU(),

      nn.Linear(1000, 101),
      nn.Softmax(dim = 1) #apply soft max to the second dimension, ignoring batch
    )
    print(f"Classifier layers: {self.layers}")

  def forward(self, x):
    x = self.layers(x)
    return x

def train_model(model, trainset, epoch_num, criterion, optimizer):
  start_time = time.time()
  for epoch in range(epoch_num):
    model.train()
    print(str(epoch + 1) + "th epoch")
    for batch in trainset:
      labels, inputs = batch
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()

      optimizer.step()
    print(f"Accuracy at {epoch + 1}th epoch: {assess_accuracy(model, testset)}")

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time: {int(elapsed_time)} seconds")

def assess_accuracy(model, testloader):
  model.eval()
  predictions = torch.empty(0).to(device)
  actual_labels = torch.empty(0).to(device)
  with torch.no_grad():
    for data in testset:
      labels, images  = data
      labels = labels.to(device)
      actual_labels = torch.cat((actual_labels, labels), dim = 0)

      images = images.to(device)
      outputs = model(images)
      predictions = torch.cat((predictions, outputs), dim = 0)

    metric = MulticlassAccuracy()
    metric.update(predictions, actual_labels)
    return metric.compute()

#                                                       TRAINING THE NN
if __name__ == "__main__":
  start_time = time.time()

  batch_size = 32
  print(f"Batch size: {batch_size}")

  trainset, ft_concat_size = frozen_features.data_set_from_csv("doublef101_aug.csv", batch_size)

  testset, _ = frozen_features.data_set_from_csv("doublef101_test.csv", batch_size)

  end_time = time.time()
  elapsed_time = end_time - start_time

  print(f"Elapsed time for csv dataset reading: {int(elapsed_time)} seconds")

  classifier = Food101kClassifier().to(device)
  criterion = nn.CrossEntropyLoss()
  epoch_num = 50
  learning_rate = 0.001
  optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
  train_model(classifier, trainset, epoch_num, criterion, optimizer)
