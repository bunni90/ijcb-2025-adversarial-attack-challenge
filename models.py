import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import kornia.augmentation as K
import kornia

class Resnet18MoreThanRGB(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18MoreThanRGB, self).__init__()
        self.resnet = torchvision.models.resnet18(num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=9*3+1,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None
        )

    def forward(self, x):
        x = torch.clamp(x, 0., 1.)
        x_aug = torch.cat([
            x, 
            kornia.enhance.jpeg_codec_differentiable(x, torch.tensor([80.0]).to(x.device)),
            kornia.enhance.jpeg_codec_differentiable(x, torch.tensor([70.0]).to(x.device)),
            kornia.enhance.jpeg_codec_differentiable(x, torch.tensor([50.0]).to(x.device)),
            kornia.color.rgb_to_grayscale(x),
            kornia.enhance.equalize(x),
            # kornia.filters.median_blur(x, 3),
            kornia.filters.gaussian_blur2d(x, (3, 3), (1.5, 1.5)),
            kornia.morphology.closing(x, kernel=torch.ones(5, 5).to(x.device)),
            kornia.morphology.opening(x, kernel=torch.ones(5, 5).to(x.device)),
            kornia.geometry.transform.scale(x, torch.tensor([[1.1, 1.1]]).to(x.device)),
        ], 1)
        return self.resnet(x_aug)
        
class ModelMultiToBinaryWrapper(nn.Module):
    def __init__(self, model, num_input_classes):
        super(ModelMultiToBinaryWrapper, self).__init__()
        self.model = model
        self.image_size = model.image_size
        self.linear = nn.Linear(num_input_classes, 2)

    def forward(self, x, return_binary_logits=False):
        multi_logits = self.model(x)
        if return_binary_logits:
            binary_logits = self.linear(multi_logits.detach())
            return multi_logits, binary_logits
        else:
            return multi_logits
