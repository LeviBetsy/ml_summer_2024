
import torch
import pandas
import os
import torchvision
import time
from utils.food5kDataset import Food5kDataset
from utils.pretrained import alexfc6_vgg16fc6
from torchvision.models import AlexNet_Weights

dataset_ratio = 1

device_name = "mps"
device = torch.device(device_name)

#                                                       DATASET

transforms = AlexNet_Weights.IMAGENET1K_V1.transforms()

# FOOD 5K
#trainset has 3000 images
trainset5k = Food5kDataset("train", transforms)
trainset5k, _ = torch.utils.data.random_split(trainset5k, [dataset_ratio, 1 - dataset_ratio])
trainloader5k = torch.utils.data.DataLoader(trainset5k, batch_size=1, shuffle=True)

testset5k = Food5kDataset("test", transforms)
testset5k, _ = torch.utils.data.random_split(testset5k, [dataset_ratio, 1 - dataset_ratio])
testloader5k = torch.utils.data.DataLoader(testset5k, batch_size=1, shuffle=False)

# FOOD 101K
trainset = torchvision.datasets.Food101(root='./data', split="train", download=True, transform=transforms)
trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

testset = torchvision.datasets.Food101(root='./data', split="test",
                                       download=True, transform=transforms)
testset, _ = torch.utils.data.random_split(testset, [dataset_ratio, 1 - dataset_ratio])
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

def extract_frozen_features(out_file, dataloader):
  pretrained_models = alexfc6_vgg16fc6()
  with torch.no_grad():
    # ft_set = torch.empty(0).to(device)
    ft_set = torch.empty(0)
    for index, data in enumerate(dataloader):
      img, label = data
      img = img.to(device)
      label = label.to(device)

      label_w_ft = label
      for pretrained_model in pretrained_models:
        
        ft = pretrained_model(img)

        #turn features tensor into a tensor with label at beginning
        label_w_ft = torch.cat((label_w_ft, ft.squeeze(0)), dim = 0).to(device)
      # print(label_w_ft.shape)
      # ft_set = torch.cat((ft_set, label_w_ft.unsqueeze(0)), dim = 0).to(device)
      ft_set = torch.cat((ft_set, label_w_ft.unsqueeze(0).cpu()), dim = 0)

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
  # extract_frozen_features("food5ktest.csv", testloader)
  extract_frozen_features("test.csv", trainloader5k)
  # extract_frozen_features("food101k_10000images.csv", trainloader)
  print("finished")
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Extraction time: {int(elapsed_time)} seconds")
  