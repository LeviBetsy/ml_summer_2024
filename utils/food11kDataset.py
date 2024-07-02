import os
import torch
from PIL import Image
import torchvision
import pandas as pd
import matplotlib.pyplot as plt



class Food11kDataset(torch.utils.data.Dataset):
  def __init__(self, split="train", transform=None):
    if (split == "train"):
      #train has 9866 images
      self.img_dir = "data/food11k/training"
    elif (split == "test"):
      #eval has 3347 images
      self.img_dir = "data/food11k/evaluation"

    self.annotations = pd.read_csv(os.path.join(self.img_dir, "food11k.csv"))
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    # for i in range(len(self.annotations)):
    #   print(self.annotations.loc[i])
    img_path = self.annotations.loc[idx][0]

      
    # image = read_image(img_path, torchvision.io.ImageReadMode.RGB)
    image = Image.open(img_path)
    if self.transform:
      image = self.transform(image)
    return image, int(self.annotations.loc[idx][1])

if __name__ == "__main__":
  dataset = Food11kDataset("train", None)
  print(dataset[0])
  print(len(dataset))

  # #show some image
  figure = plt.figure(figsize=(8, 8))
  cols, rows = 2, 5
  # indx = [0,1500, 1, 1501, 2, 1502, 3, 1503, 4, 1504]
  for i in range(cols * rows):
      img, label = dataset[i]
      img = img.permute(1, 2, 0)
      figure.add_subplot(rows, cols, i + 1)
      plt.title(label)
      plt.axis("off")
      plt.imshow(img.squeeze(), cmap="gray")
  plt.show()
  