import torch
import os
import torch.nn as nn
import time
import torch.optim as optim
import torchvision
from torcheval.metrics import MulticlassAccuracy
import frozen_features
import random
from sklearn.decomposition import PCA
import numpy as np

from food_classifier import train_model, assess_accuracy

device_name = "mps"
device = torch.device(device_name)

class Food101AdjustableClassifier(nn.Module):
  def __init__(self, hp): #hp stands for hyperparameters
    super().__init__()
    d1, d2 = hp
    self.layers = nn.Sequential(
      nn.ReLU(),
      nn.Linear(ft_concat_size, d1),
      nn.BatchNorm1d(d1),
      nn.ReLU(),

      nn.Linear(d1, d2),
      nn.BatchNorm1d(d2),
      nn.ReLU(),

      nn.Linear(d2, 101), #TODO: DONT FORGET TO CHANGE TO 101
      nn.Softmax(dim = 1) #apply soft max to the second dimension, ignoring batch
    )
    # print(f"Classifier layers: {self.layers}")

  def forward(self, x):
    x = self.layers(x)
    return x

class Food101AdjustablePCAClassifier(nn.Module):
  def __init__(self, hp): #hp stands for hyperparameters
    super().__init__()
    d1 = hp[0]
    self.layers = nn.Sequential(
      nn.Linear(number_of_components, d1),
      nn.BatchNorm1d(d1),
      nn.ReLU(),

      nn.Linear(d1, d2),
      nn.BatchNorm1d(d2),
      nn.ReLU(),

      nn.Linear(d2, 101), #TODO: DONT FORGET TO CHANGE TO 101
      nn.Softmax(dim = 1) #apply soft max to the second dimension, ignoring batch
    )
    # print(f"Classifier layers: {self.layers}")

  def forward(self, x):
    #convert to numpy for pca
    x = x.cpu().numpy() #send to cpu for numpy conversion
    x = pca.transform(x) #reshape into shape of (1, number of features -1 is self-infer size of original)
    x = torch.from_numpy(x) #convert back to tensor
    x = x.to(device) #send back to gpu for process
    x = self.layers(x)
    return x

class SwarmOptimize():
  def __init__(self, bounds, trainset, epoch_num, criterion, testset):
    self.trainset = trainset
    self.epoch_num = epoch_num
    self.criterion = criterion
    self.testset = testset
    self.bounds = bounds

    self.hyperparam_num = len(bounds)

    self.best_hp = [round(random.uniform(tup[0], tup[1])) for tup in self.bounds]
    print(f"Initial hyperparameters: {self.best_hp}")
    self.best_model = Food101AdjustablePCAClassifier(self.best_hp).to(device)

    optimizer = optim.Adam(self.best_model.parameters(), lr=0.001)
    train_model(self.best_model, trainset, epoch_num, criterion, optimizer, testset, verbose=False)

    self.best_accuracy = assess_accuracy(self.best_model, testset)
    print(f"Initial accuracy: {self.best_accuracy}")

    #search space
    self.A = [tup[1] - tup[0] for tup in self.bounds]


  
  #returns a tuple of (hyperparameters, accuracy) for a model taken from the search space
  def generate_model(self):
    #hyper param
    hp = [round(random.uniform(self.best_hp[i] - self.A[i], self.best_hp[i] + self.A[i])) for i in range(self.hyperparam_num)]
    for i in range (self.hyperparam_num):
      while hp[i] == self.best_hp[i]:
        hp[i] = round(random.uniform(self.best_hp[i] - self.A[i], self.best_hp[i] + self.A[i]))
    hp = [hp[i] if hp[i] >= self.bounds[i][0] else self.bounds[i][0] for i in range(self.hyperparam_num)]
    hp = [hp[i] if hp[i] <= self.bounds[i][1] else self.bounds[i][1] for i in range(self.hyperparam_num)]

    model = Food101AdjustablePCAClassifier(hp).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, self.trainset, self.epoch_num, self.criterion, optimizer, self.testset, verbose=False)

    return hp, assess_accuracy(model, self.testset)
  
  def swarm_optimization(self, Ca, Cr, iteration, num_spark):
    for i in range(iteration):
      sparks = []
      for s in range(num_spark):
        sparks.append(self.generate_model())

      spark_hp, spark_accuracy = sparks[0]
      #find the best accuracy model from the sparks
      for spark in sparks[1:]:
        if spark[1] > spark_accuracy:
          spark_hp, spark_accuracy = spark

      #decrease search space
      if spark_accuracy > self.best_accuracy:
        print(f"Has a better spark: {spark_accuracy} > {self.best_accuracy}")
        print(f"Better hyper param: {spark_hp}")
        self.best_hp = spark_hp
        self.best_accuracy = spark_accuracy
        self.A = [d*Ca for d in self.A]
      else:
        print("Narrowing search space because did not find better spark")
        self.A = [d*Cr for d in self.A]

    print(f"After {iteration} iterations, Ca {Ca}, Cr {Cr}, number of sparks per iteration {num_spark}:"\
      f"best hyperparameters: {self.best_hp}, accuracy: {self.best_accuracy}"
    )


#                                                       TRAINING THE NN
if __name__ == "__main__":
  # EXTRACTING CSV

  batch_size = 32
  print(f"Batch size: {batch_size}")

  trainset, ft_concat_size = frozen_features.data_set_from_csv("food101ktrain.csv", batch_size)
  testset, _ = frozen_features.data_set_from_csv("food101ktest.csv", batch_size)

  #PCA

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

  # MODELS

  # concat_size (8000 something) -> d1 -> d2 -> 101
  search_range = [(250, 3000), (250, 3000)]

  swarm = SwarmOptimize(search_range, trainset, epoch_num = 4, criterion = nn.CrossEntropyLoss(), testset = testset)

  swarm.swarm_optimization(1.3, 0.7, 5, 5)
  # print(swarm_optimization.generate_model())
