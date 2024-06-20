import torchvision.transforms as transforms
import torchvision
import torch

transforms = transforms.Compose([

  transforms.RandomHorizontalFlip(p=0.5),
  transforms.ToTensor()

])

trainset = torchvision.datasets.Food101(root='./data', split="train",
                                        download=True, transform=transforms)

img = trainset[0]
print(len(trainset))
# trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

