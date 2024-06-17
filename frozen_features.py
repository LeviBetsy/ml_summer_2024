
import torch
import pandas
import os
import torchvision
from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights
from torchvision.io import read_image
import torch.nn as nn
import time
import torchvision.transforms.v2 as v2

dataset_ratio = 0.005

device_name = "mps"
device = torch.device(device_name)

#                                                             PREPARING MODELS

alexnet_model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)

vgg16_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

#freezing
alexnet_model.eval()
for param in alexnet_model.parameters():
  param.requires_grad = False

vgg16_model.eval()
for param in vgg16_model.parameters():
  param.requires_grad = False

class AlexNetFc6(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.features = nn.Sequential(
        alexnet_model.features,
        alexnet_model.avgpool
    )
    self.fc6 = alexnet_model.classifier[:2] #slice the module to only stop until fc 6
    # print(self.fc6)

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
    # print(self.fc6)

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

pretrained_models = [alexnetfc6, vgg16fc6]

#                                                       DATASET

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


transforms = AlexNet_Weights.IMAGENET1K_V1.transforms()
# print(torchvision.transforms.AutoAugmentPolicy.IMAGENET)
# transforms = v2.AutoAugment()

#training data
#trainset has 3000 images
trainset = Food5kDataset("train", transforms)
trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

testset = Food5kDataset("test", transforms)
testset, _ = torch.utils.data.random_split(testset, [dataset_ratio, 1 - dataset_ratio])
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# food 101k

trainset = torchvision.datasets.Food101(root='./data', split="train",
                                        download=True, transform=transforms)
trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

print(len(trainset))


testset = torchvision.datasets.Food101(root='./data', split="test",
                                       download=True, transform=transforms)
testset, _ = torch.utils.data.random_split(testset, [dataset_ratio, 1 - dataset_ratio])
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

def extract_frozen_features(out_file, dataloader):
  with torch.no_grad():
    # ft_set = torch.empty(0).to(device)
    ft_set = torch.empty(0)
    for _, data in enumerate(dataloader):
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

    if index % 1000 == 0:
      print(f"{index}th image")
  return training_dataset, ft_concat_size


if __name__ == "__main__":
  start_time = time.time()
  # extract_frozen_features("food5ktest.csv", testloader)
  # extract_frozen_features("food5ktrain.csv", trainloader)
  extract_frozen_features("food101ktrain_aug.csv", trainloader)
  print("finished")
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Extraction time: {int(elapsed_time)} seconds")
  