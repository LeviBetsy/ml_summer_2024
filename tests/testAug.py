import os
import torch
from torchvision.io import read_image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

aug = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((227,227)),
  transforms.ColorJitter(brightness=0.5),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor()
])


class TestSet(torch.utils.data.Dataset):
  def __init__(self, transform=None):

    self.img_dir = "data/testaug"
    #transform
    self.transform = transform

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, "0.jpg")
    label = 1
    image = read_image(img_path, torchvision.io.ImageReadMode.RGB)
    # print(image[0][0])
    if self.transform:
      image = self.transform(image)
    return image, label

dataset = TestSet(aug)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for epoch in range(5):
  for data, _ in dataloader:
    img = data[0].permute(1, 2, 0)
    plt.imshow(img)
    plt.show()
  
#it does in fact does permutation everytime you do for data, _ in dataloader
#thus the csv lacks that functionality since it has always done the same thing for one iteration. We can stack it.