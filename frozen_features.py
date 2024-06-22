
import torch
import pandas
import os
import torchvision
import time
import torchvision.transforms as transforms
from utils.food5kDataset import Food5kDataset
from utils.pretrained import alexfc6_vgg16fc6, alexnetfc6, vgg16fc6
from torchvision.models import AlexNet_Weights
from torchvision.io import read_image

import matplotlib.pyplot as plt

dataset_ratio = 1

device_name = "mps"
device = torch.device(device_name)

def extract_frozen_features(out_file, dataloader):
  # pretrained_models = alexfc6_vgg16fc6()
  alex = alexnetfc6
  vgg = vgg16fc6
  with torch.no_grad():
    ft_set = torch.empty(0)
    for index, data in enumerate(dataloader):
      img, label = data
      img = img.to(device)
      label = label.to(device)

      label_w_ft = label

      vgg_ft = vgg(img)
      label_w_ft = torch.cat((label_w_ft, vgg_ft.squeeze(0)), dim = 0).to(device)


      aleximg = transforms.functional.resize(img, size=(227,227))

      alex_ft = alex(aleximg)
      label_w_ft = torch.cat((label_w_ft, alex_ft.squeeze(0)), dim = 0).to(device)

      # for pretrained_model in pretrained_models:
        
      #   ft = pretrained_model(img)

      #   #turn features tensor into a tensor with label at beginning
      #   label_w_ft = torch.cat((label_w_ft, ft.squeeze(0)), dim = 0).to(device)

      # print(label_w_ft.shape)
      # ft_set = torch.cat((ft_set, label_w_ft.unsqueeze(0)), dim = 0).to(device)
      ft_set = torch.cat((ft_set, label_w_ft.unsqueeze(0)), dim = 0)

      if index % 1000 == 0:
        print(f"{index}th image")

    ft_set = ft_set.cpu().numpy()
    df = pandas.DataFrame(ft_set)
    df.to_csv("frozen_features/" + out_file, index=False)

def data_set_from_csv(csv_file, batch_size):
  #NOTE: data[0] is the first data line, does not include header
  data = pandas.read_csv('frozen_features/' + csv_file)
  #data shape is (num_images, 1 label + num of features)

  ft_concat_size = data.shape[1] - 1
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

    
  return training_dataset, ft_concat_size


if __name__ == "__main__":
  
  start_time = time.time()
  #                                                       DATASET
  data_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomResizedCrop(size=(224,224), scale=(0.4,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5458, 0.4443, 0.3442), (0.2711, 0.2740, 0.2792)),
  ])

  test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5458, 0.4443, 0.3442), (0.2711, 0.2740, 0.2792)),
  ])

  print(f"Apply data augmentation: {data_aug}")

  # FOOD 101K
  trainset = torchvision.datasets.Food101(root='./data', split="train", download=True, transform=data_aug)

  bigTrainSet = torch.utils.data.ConcatDataset([trainset, trainset])
  # print(len(bigTrainSet))
  trainloader = torch.utils.data.DataLoader(bigTrainSet, batch_size=1, shuffle=True)

  testset = torchvision.datasets.Food101(root='./data', split="test", download=True, transform=test_transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

  # extract_frozen_features("test.csv", trainloader5k)

  extract_frozen_features("doublef101_test.csv", testloader)
  # extract_frozen_features("food101k_10000images.csv", trainloader)
  print("finished")
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Extraction time: {int(elapsed_time)} seconds")
  