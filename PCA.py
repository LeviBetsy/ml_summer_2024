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
from sklearn.decomposition import PCA

device_name = "mps"
device = torch.device(device_name)

start_time = time.time()

batch_size = 32
print(f"batch size: {batch_size}")

# trainset, ft_concat_size = frozen_features.data_set_from_csv("food5ktrain.csv", batch_size)
# print(f"Number of batches: {len(trainset)}")

# testset, _ = frozen_features.data_set_from_csv("food5ktest.csv", batch_size)

trainset, ft_concat_size = frozen_features.data_set_from_csv("food101ktrain.csv", batch_size)
# trainset, ft_concat_size = frozen_features.data_set_from_csv("food101ktrain.csv", batch_size)
print(f"Number of batches: {len(trainset)}")

testset, _ = frozen_features.data_set_from_csv("food101ktest.csv", batch_size)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for csv dataset reading: {int(elapsed_time)} seconds")

start_time = time.time()


#unbatch, so they're all one list of feature tensors and turn each tensor into numpy array
unbatched_features = [f.numpy() for (labels, features) in trainset for f in features]

print(f"Number of samples for PCA {len(unbatched_features)}")

#converting trainset into numpy array of shape (num_of_images, num_of_features)
X = np.array(unbatched_features)

pca = PCA(n_components=2000, random_state=42)
pca.fit(X)
number_of_components = pca.n_components_

print("Number of features (PCs) used:", pca.n_components_)

# Optional: Print explained variance ratio per PC
print("Explained variance ratio per component:")
print(pca.explained_variance_ratio_)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for PCA construction: {int(elapsed_time)} seconds")





#                           NN


class Food101kClassifier(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.layers = nn.Sequential(
      # nn.ReLU(),
      nn.Linear(number_of_components, 500),
      # nn.Linear(1, 500),
      nn.BatchNorm1d(500),
      nn.ReLU(),
      nn.Linear(500, 256),
      
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(256, 101),
      # nn.BatchNorm1d(256),
      # nn.ReLU(),
      # nn.Linear(256, 101),

      nn.Softmax(dim = 1) #apply soft max to the second dimension, ignoring batch
    )
    print(f"Classifier layers: {self.layers}")

  def forward(self, x):
    
    #convert to numpy for pca
    x = x.cpu().numpy() #send to cpu for numpy conversion
    x = pca.transform(x) #reshape into shape of (1, number of features -1 is self-infer size of original)
    x = torch.from_numpy(x) #convert back to tensor
    x = x.to(device) #send back to gpu for process
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
      # print(labels.shape)
      actual_labels = torch.cat((actual_labels, labels), dim = 0)

      images = images.to(device)
      outputs = model(images)
      predictions = torch.cat((predictions, outputs), dim = 0)

    metric = MulticlassAccuracy()
    metric.update(predictions, actual_labels)
    return metric.compute()

#                                                       TRAINING THE NN
if __name__ == "__main__":
  classifier = Food101kClassifier().to(device)
  criterion = nn.CrossEntropyLoss()
  epoch_num = 20
  learning_rate = 0.001
  optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
  train_model(classifier, trainset, epoch_num, criterion, optimizer)
