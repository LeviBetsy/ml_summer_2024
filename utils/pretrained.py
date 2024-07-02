import torchvision.models
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

class FrozenAlexNetFc6(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    #freezing
    alexnet_model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)
    alexnet_model.eval()
    for param in alexnet_model.parameters():
      param.requires_grad = False

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

class FrozenVGG16Fc6(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    #freezing
    vgg16_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)
    vgg16_model.eval()
    for param in vgg16_model.parameters():
      param.requires_grad = False

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

class FrozenVGG16Conv(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    #freezing
    vgg16_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)
    vgg16_model.eval()
    for param in vgg16_model.parameters():
      param.requires_grad = False

    self.features = vgg16_model.features
    self.avgpool = vgg16_model.avgpool
    # self.fc6 = vgg16_model.classifier[:1] #slice the module to only stop until fc 6
    # print(self.fc6)

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x

class FrozenResnetConv(nn.Module):
  def __init__(self, resnet_name): #hp stands for hyperparameters
    super().__init__()
    if resnet_name == "resnet18":
      resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
    elif resnet_name == "resnet34":
      resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights='ResNet34_Weights.IMAGENET1K_V1')
    elif resnet_name == "resnet50":
      resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
    else:
      raise Exception("Can't find resnet module")

    resnet.eval()
    for param in resnet.parameters():
      param.requires_grad = False

    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1 
    self.relu = resnet.relu 
    self.maxpool = resnet.maxpool 
    self.layer1 = resnet.layer1 
    self.layer2 = resnet.layer2 
    self.layer3 = resnet.layer3
    self.layer4 = resnet.layer4
    self.avgpool = resnet.avgpool


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x

class FrozenEfficientNetConv(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    efnet = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
    self.features = efnet.features
    self.avgpool = efnet.avgpool
  
    efnet.eval()
    for param in efnet.parameters():
      param.requires_grad = False

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x

class UnfrozenEfficientNetConv(nn.Module):
  def __init__(self): #hp stands for hyperparameters
    super().__init__()
    self.efnet = FrozenEfficientNetConv()
    for name, conv_layer in self.efnet.features.named_children():
      if name in ["7", "8"]:
        conv_layer.requires_grad = True
  
  def forward(self, x):
    x = self.efnet(x)
    return x

device_name = "mps"
device = torch.device(device_name)

if __name__ == "__main__":
  efnet = UnfrozenEfficientNetConv()
    