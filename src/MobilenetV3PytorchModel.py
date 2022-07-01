from torch import nn
import torchvision.models as models

class MobilenetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # Заменяем последние 2 полносвязных слоя на имеющие нужную размерность
        self.backbone.classifier[0] = nn.Linear(576, 256)
        self.backbone.classifier[3] = nn.Linear(256, 1)

        
    def get_layers_names(self):
         return dict(self.backbone.named_modules())

        
    def forward(self, x):
        x = self.backbone(x)
        return x