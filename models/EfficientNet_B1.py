
import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNet_B1(nn.Module):
    def __init__(self, n_classes=2):
        super(EfficientNet_B1, self).__init__()
        self.features = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2
        )
        self.classifier = nn.Linear(1000, n_classes, bias=True)

    @torch.jit.export
    def get_latent_vector(self, x):
        x = self.features(x)
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
