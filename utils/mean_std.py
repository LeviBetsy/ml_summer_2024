import torchvision.transforms as transforms
import torchvision
import torch

resize = transforms.Compose([
  # transforms.Resize((227,227)),
  transforms.ToTensor()
])

print(f"Apply resize: {resize}")

trainset = torchvision.datasets.Food101(root='./data', split="train",
                                        download=True, transform=resize)
testset = torchvision.datasets.Food101(root='./data', split="test",
                                        download=True, transform=resize)
concat_set = torch.utils.data.ConcatDataset([trainset, testset])

mean = torch.tensor([0.0, 0.0, 0.0])
squared = torch.tensor([0.0, 0.0, 0.0])
for img, _ in concat_set:
  # print(img.shape)
  mean += torch.mean(img, dim= [1,2])
  squared += torch.mean(img ** 2, dim= [1,2])
mean = mean/len(concat_set)
std =  (squared/len(concat_set) - mean**2) ** 0.5

print(f"Mean: {mean}")
print(f"Std: {std}")

# resize = transforms.Compose([
#   transforms.Resize((224,224)),
#   transforms.ToTensor()
# ])

# print(f"Apply resize: {resize}")

# trainset = torchvision.datasets.Food101(root='./data', split="train",
#                                         download=True, transform=resize)
# testset = torchvision.datasets.Food101(root='./data', split="test",
#                                         download=True, transform=resize)
# concat_set = torch.utils.data.ConcatDataset([trainset, testset])

# mean = torch.tensor([0.0, 0.0, 0.0])
# squared = torch.tensor([0.0, 0.0, 0.0])
# for img, _ in concat_set:
#   # print(img.shape)
#   mean += torch.mean(img, dim= [1,2])
#   squared += torch.mean(img ** 2, dim= [1,2])
# mean = mean/len(concat_set)
# std =  (squared/len(concat_set) - mean**2) ** 0.5

# print(f"Mean: {mean}")
# print(f"Std: {std}")


# #for Levi understanding of wild torch.mean
# #basically just calculating the mean at a layer, mean can be calculated using the nested tensors too, resulting in another tensor
# pic = torch.tensor([[[1, 0], [0, 0]],[[0, 0],[0, 0]],[[0, 0],[0, 0]]], dtype=torch.float32) #shape : (3, 2, 2)
# print(pic.shape)

# print(torch.mean(pic, dim= [1,2])) # find mean of 2, then mean of 1
# print(torch.mean(pic, dim= 0)) #shape (2,2) because that is the thing inside of each things at dimension 0
# print(torch.mean(pic, dim= 1))
# print(torch.mean(pic, dim= 2)) #shape (3,2)
# print(torch.mean(pic, dim= [0,2]))







# trainset, _ = torch.utils.data.random_split(trainset, [dataset_ratio, 1 - dataset_ratio])
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

