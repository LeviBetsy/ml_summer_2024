from torchvision.transforms import v2
import torchvision
import torch


trainset = torchvision.datasets.Food101(root='./data', split="train",
                                        download=True, transform=None)
print(len(trainset))

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.Food101(root='./data', split="train",
                                        download=True, transform=transforms)

img = trainset[0]
print(len(trainset))
# trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

