from torchvision.models import alexnet, AlexNet_Weights, vgg16, VGG16_Weights
import torch
import torch.nn as nn

class AlexNetFc6(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    #freezing
    alexnet_model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)

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
    #freezing
    vgg16_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

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

device_name = "mps"
device = torch.device(device_name)

alexnetfc6 = AlexNetFc6().to(device)
alexnetfc6.eval()
for param in alexnetfc6.parameters():
  param.requires_grad = False

vgg16fc6 = VGG16Fc6().to(device)
vgg16fc6.eval()
for param in vgg16fc6.parameters():
  param.requires_grad = False

    
def alexfc6_vgg16fc6():
  device_name = "mps"
  device = torch.device(device_name)

  alexnetfc6 = AlexNetFc6().to(device)
  alexnetfc6.eval()
  for param in alexnetfc6.parameters():
      param.requires_grad = False

  vgg16fc6 = VGG16Fc6().to(device)
  vgg16fc6.eval()
  for param in vgg16fc6.parameters():
    param.requires_grad = False

  return [alexnetfc6, vgg16fc6]