import torch
import os
import torch.nn as nn
import time
import torch.optim as optim
import torchvision
from torcheval.metrics import MulticlassAccuracy
from torchvision.models import AlexNet_Weights
import utils.pretrained as pretrained
from food_classifier import assess_accuracy
from utils.food11kDataset import Food11kDataset

device_name = "mps"
device = torch.device(device_name)


#                                         NN

class Food11kClassifier(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.efnet = pretrained.UnfrozenEfficientNetConv()
    self.layers = nn.Sequential(
      nn.ReLU(),
      nn.Linear(1280, 1000),
      nn.BatchNorm1d(1000),
      nn.ReLU(),
      nn.Linear(1000, 400),
      nn.BatchNorm1d(400),
      nn.ReLU(),
      nn.Linear(400, 11),
      nn.Softmax(dim = 1) #apply soft max to the second dimension, ignoring batch
    )
    print(f"Classifier layers: {self.layers}")

  def forward(self, x):
    x = self.efnet(x)
    x = self.layers(x)
    return x

def train_model(model, trainset, epoch_num, criterion, optimizer, testset, verbose):
  start_time = time.time()
  for epoch in range(epoch_num):
    model.train()
    for name, conv_layer in model.efnet.efnet.features.named_children():
      if name not in ["7", "8"]:
        conv_layer.eval()
    if verbose:
      print(str(epoch + 1) + "th epoch")
    for batch in trainset:
      inputs, labels = batch
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()

      optimizer.step()
    if verbose:
      print(f"Accuracy at {epoch + 1}th epoch: {assess_accuracy(model, testset)}")

  end_time = time.time()
  elapsed_time = end_time - start_time
  # if verbose:
  print(f"Elapsed time: {int(elapsed_time)} seconds")


#                                                       TRAINING THE NN
if __name__ == "__main__":
  #food 11
  batch_size = 32
  print(f"Batch size: {batch_size}")
  print("unfroze 7, 8")

  trainset = Food11kDataset("train", transform=AlexNet_Weights.IMAGENET1K_V1.transforms())
  testset = Food11kDataset("test", transform=AlexNet_Weights.IMAGENET1K_V1.transforms())

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

  classifier = Food11kClassifier().to(device)
  criterion = nn.CrossEntropyLoss()
  epoch_num = 50
  learning_rate = 0.001
  optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
  train_model(classifier, trainloader, epoch_num, criterion, optimizer, testloader, True)
