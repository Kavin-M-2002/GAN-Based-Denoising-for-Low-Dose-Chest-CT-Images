import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=16):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:layer_index])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
