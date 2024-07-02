#USELESS, JUST NORMALIZE TO 0-1


import torchvision.transforms as transforms
import torchvision
import torch
from food11kDataset import Food11kDataset
from torchvision.models import AlexNet_Weights

# resize = transforms.Compose([
#   # transforms.Resize((227,227)),
#   # transforms.ToTensor()
# ])

# print(f"Apply resize: {resize}")

trainset = Food11kDataset("train", transforms.ToTensor())
testset = Food11kDataset("test", transforms.ToTensor())

concat_set = torch.utils.data.ConcatDataset([trainset, testset])

mean = torch.tensor([0.0, 0.0, 0.0])
squared = torch.tensor([0.0, 0.0, 0.0])
for img, _ in concat_set:
  fl_img = img.double()
  mean += torch.mean(fl_img, dim= [1,2])
  squared += torch.mean(fl_img ** 2, dim= [1,2])
mean = mean/len(concat_set)
std =  (squared/len(concat_set) - mean**2) ** 0.5

print(f"Mean: {mean}")
print(f"Std: {std}")


# # #for Levi understanding of wild torch.mean
# # #basically just calculating the mean at a layer, mean can be calculated using the nested tensors too, resulting in another tensor
# # pic = torch.tensor([[[1, 0], [0, 0]],[[0, 0],[0, 0]],[[0, 0],[0, 0]]], dtype=torch.float32) #shape : (3, 2, 2)
# # print(pic.shape)

# # print(torch.mean(pic, dim= [1,2])) # find mean of 2, then mean of 1
# # print(torch.mean(pic, dim= 0)) #shape (2,2) because that is the thing inside of each things at dimension 0
# # print(torch.mean(pic, dim= 1))
# # print(torch.mean(pic, dim= 2)) #shape (3,2)
# # print(torch.mean(pic, dim= [0,2]))