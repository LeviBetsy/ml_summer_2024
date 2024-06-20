import os
import torch
from torchvision.io import read_image
import torchvision

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