import torch
import torchvision
from torch import nn


class ImageClfModel(nn.Module):
    def __init__(
        self, in_channel=1, num_classes=9, backbone="resnet18", pretrained=False
    ):
        super(ImageClfModel, self).__init__()
        # Handle pretrained weights for torchvision >= 0.13
        if "mobilenet_v3" in backbone:
            weights = (
                torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
                if pretrained and backbone == "mobilenet_v3_large"
                else torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                if pretrained and backbone == "mobilenet_v3_small"
                else None
            )
            self.backbone = getattr(torchvision.models, backbone)(weights=weights)
            # Change input channels if needed
            if in_channel != 3:
                first_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(
                    in_channel,
                    first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None,
                )
            # Change classifier
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            # For ResNet and similar
            self.backbone = getattr(torchvision.models, backbone)(pretrained=pretrained)
            if in_channel != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channel,
                    self.backbone.conv1.out_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        if num_classes == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return logits, self.final_activation(logits)


if __name__ == "__main__":
    # Example usage
    model = ImageClfModel(in_channel=1, num_classes=9, backbone="resnet18")
    print(model)
    # Test with a random input
    x = torch.randn(1, 1, 224, 224)  # Batch size of 1, single channel input
    logits, probabilities = model(x)
    print(logits.shape, probabilities.shape)
