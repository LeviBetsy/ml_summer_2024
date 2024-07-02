
import torch
import pandas
import os
import torchvision
import time
from torchvision.transforms import v2 as transforms
from utils.food5kDataset import Food5kDataset
from utils.food11kDataset import Food11kDataset

import utils.pretrained as pretrained
from torchvision.models import AlexNet_Weights
from torchvision.io import read_image

import matplotlib.pyplot as plt

dataset_ratio = 1

device_name = "mps"
device = torch.device(device_name)

def extract_frozen_features(pretrained_model, out_file, dataloader):
  # alex = alexnetfc6
  # vgg = vgg16fc6
  pretrained = pretrained_model.to(device)

  with torch.no_grad():
    ft_set = torch.empty(0)
    for index, data in enumerate(dataloader):
      img, label = data
      img = img.to(device)
      label = label.to(device)

      label_w_ft = label

      #Resnet
      ft = pretrained(img)
      label_w_ft = torch.cat((label_w_ft, ft.squeeze(0)), dim = 0).to(device) 

      ft_set = torch.cat((ft_set, label_w_ft.cpu().unsqueeze(0)), dim = 0)
      if index % 1000 == 0:
        print(f"{index}th image")

    ft_set = ft_set.cpu().numpy()
    df = pandas.DataFrame(ft_set)
    df.to_csv("frozen_features/" + out_file, index=False)

def data_set_from_csv(csv_file, batch_size):
  start_time = time.time()

  #NOTE: data[0] is the first data line, does not include header
  data = pandas.read_csv('frozen_features/' + csv_file)
  #data shape is (num_images, 1 label + num of features)

  ft_concat_size = data.shape[1] - 1 
  # ft_concat_size = (data.shape[1] - 1) // 2 #TODO: CHANGE
  print(f"Number of input features for one image: {ft_concat_size}")

  training_dataset = []
  index = 0
  while index < data.shape[0]:
    labels_tensor = []
    features_tensor = torch.empty(0)
    for _ in range(batch_size):
      if index >= data.shape[0]: #handle last batch to include leftovers
        break
      
      #EXTRACTING LABELS
      label = int(data.iloc[index][0])
      labels_tensor.append(label)

      #EXTRACTING FEATURES
      features = data.iloc[index][1:] 
      #convert from Dataframe to Numpy array
      features = features.to_numpy()
      #convert from Numpy array to tensor
      features = torch.from_numpy(features)
      features_tensor = torch.cat((features_tensor, features.unsqueeze(0)), dim = 0)
      index = index + 1

    labels_tensor = torch.tensor(labels_tensor, dtype=torch.long)
    features_tensor = features_tensor.to(torch.float32)
    training_dataset.append((labels_tensor, features_tensor))
  
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time for csv {csv_file} dataset reading: {int(elapsed_time)} seconds")
  return training_dataset, ft_concat_size


if __name__ == "__main__":
  start_time = time.time()

  #food 11
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5565, 0.4520, 0.3452], std=[0.2717, 0.2748, 0.2816])
  ])

  print(f"Applied transform: {transform}")


  trainset = Food11kDataset("train", transform=AlexNet_Weights.IMAGENET1K_V1.transforms())
  testset = Food11kDataset("test", transform=AlexNet_Weights.IMAGENET1K_V1.transforms())

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

  csv_name = "f11/efb1conv_train.csv"
  extract_frozen_features(pretrained.FrozenEfficientNetConv(), csv_name, trainloader)
  train_time = time.time()
  print(f"Extraction time for {csv_name}: {int(time.time() - start_time)} seconds")

  csv_name = "f11/efb1conv_test.csv"
  extract_frozen_features(pretrained.FrozenEfficientNetConv(), csv_name, testloader)
  print(f"Extraction time for {csv_name}: {int(time.time() - train_time)} seconds")
  