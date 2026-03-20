import torch
from segmentation_models_pytorch.base.modules import SCSEModule
from torch import nn

from rosaid.models.base import BaseFeatureModel
from rosaid.models.squeeze_excite import SqueezeExciteConv1D, SqueezeExciteLinear


class FeatureCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list[int] = [64, 128, 256, 512, 256, 128, 64],
        output_size: int = 9,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
        kernel_size: int = 3,
    ):
        """
        Takes input at Num_packets x 1 and outputs classification logits

        """
        super().__init__()
        self.output_size = output_size
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        layers = []
        for idx, out_ch in enumerate(hidden_channels):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_ch,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if idx % 2 == 1:
                layers.append(SqueezeExciteConv1D(out_ch))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_channels = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, features]
        # x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)  # [B, C, L]
        x = nn.functional.adaptive_avg_pool1d(x, 1)  # [B, C, 1]

        logits = self.fc(x.view(x.size(0), -1))  # [B, output_size]

        return logits


class CNN1D(BaseFeatureModel):
    def __init__(
        self,
        input_size: int = 46,
        hidden_channels: list[int] = [32, 64, 128, 256, 512, 256, 128, 64, 32],
        output_size: int = 9,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
        kernel_size: int = 3,
        se_conv: bool = True,
        flatten_features: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.use_batchnorm = use_batchnorm
        self.kernel_size = kernel_size
        self.se_conv = se_conv
        self.flatten_features = flatten_features

        # Create named blocks for feature extraction at different levels
        self.blocks = nn.ModuleDict()
        in_ch = input_size
        for idx, out_ch in enumerate(hidden_channels):
            block = []
            block.append(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False
                )
            )
            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_ch))
            block.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                block.append(nn.Dropout(dropout_rate))
            if se_conv:
                block.append(
                    SqueezeExciteConv1D(out_ch, reduction=max(4, out_ch // 16))
                )
            else:
                block.append(
                    SqueezeExciteLinear(out_ch, reduction=max(4, out_ch // 16))
                )

            # Create named block for feature pyramid access
            block = nn.Sequential(*block)
            self.blocks[f"block_{idx}"] = block
            in_ch = out_ch

        # Classifier head
        # Use global max pooling instead of flatten for better features
        if flatten_features:
            self.global_pool = nn.Flatten()
            in_ch = hidden_channels[-1] * input_size
        else:
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the pooled features
            nn.Linear(in_ch, in_ch // 2, bias=False),
            nn.BatchNorm1d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_ch // 2, output_size),
        )
        if output_size == 1:
            self.softmax = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    @property
    def summary(self) -> str:
        """
        Generate a detailed summary of the CNN1D model.

        Returns:
            str: Formatted summary containing model architecture and parameters
        """
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("CNN1D Model Summary")
        summary_lines.append("=" * 80)

        # Model configuration
        summary_lines.append(f"Input Size: {self.input_size}")
        summary_lines.append(f"Hidden Channels: {self.hidden_channels}")
        summary_lines.append(f"Dropout Rate: {self.dropout_rate}")
        summary_lines.append(f"Total Blocks: {len(self.hidden_channels)}")

        # Block-wise details
        summary_lines.append("\nBlock Architecture:")
        summary_lines.append("-" * 50)
        total_params = 0

        for block_name, block in self.blocks.items():
            block_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
            total_params += block_params

            # Get input/output channels for this block
            idx = int(block_name.split("_")[1])
            in_ch = 1 if idx == 0 else self.hidden_channels[idx - 1]
            out_ch = self.hidden_channels[idx]

            summary_lines.append(
                f"{block_name:>10}: {in_ch:>3} -> {out_ch:>3} channels | "
                f"{block_params:>8,} params"
            )

        # Classifier details
        classifier_params = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )
        total_params += classifier_params

        classifier_in = self.hidden_channels[-1]  # After global pooling
        classifier_out = self.classifier[-1].out_features
        summary_lines.append(
            f"{'classifier':>10}: {classifier_in:>3} -> {classifier_out:>3}"
            f" units | {classifier_params:>8,} params"
        )

        # Parameter summary
        summary_lines.append("\nParameter Summary:")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Total Trainable Parameters: {total_params:,}")

        # Memory estimation (rough)
        param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        summary_lines.append(f"Estimated Model Size: {param_size_mb:.2f} MB")

        # Feature extraction info
        summary_lines.append("\nFeature Extraction:")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Available Blocks: {list(self.blocks.keys())}")
        summary_lines.append(f"Feature Pyramid Levels: {len(self.blocks)}")

        # Example input/output shapes
        summary_lines.append("\nExample Shapes (batch_size=1):")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Input:  [1, {self.input_size}]")
        summary_lines.append(f"After unsqueeze: [1, 1, {self.input_size}]")

        for idx, ch in enumerate(self.hidden_channels):
            summary_lines.append(f"Block {idx}: [1, {ch}, {self.input_size}]")

        # Show global max pooling effect
        last_ch = self.hidden_channels[-1]
        summary_lines.append(f"After global max pool: [1, {last_ch}, 1]")
        summary_lines.append(f"After flatten: [1, {last_ch}]")
        out_features = self.classifier[-1].out_features
        summary_lines.append(f"Output: [1, {out_features}]")

        summary_lines.append("=" * 80)

        return "\n".join(summary_lines)

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract features from all named blocks for feature pyramid training"""
        x = x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, L]

        features = {}
        for idx in range(len(self.hidden_channels)):
            block_name = f"block_{idx}"
            x = self.blocks[block_name](x)
            features[block_name] = x

        return features, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, features]
        x = x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, L]

        features, x = self.extract_features(x)

        # Apply global max pooling instead of flatten
        x = self.global_pool(x)  # [B, C, 1]
        logits = self.classifier(x)  # [B, output_size]

        return logits, self.softmax(logits)

    def get_block_names(self) -> list[str]:
        """Return list of block names for student-teacher training"""
        return [f"block_{idx}" for idx in range(len(self.hidden_channels))]

    def get_block(self, block_name: str) -> nn.Module:
        """Get specific block by name for feature matching"""
        return self.blocks[block_name]


class BlockCNN2D(BaseFeatureModel):
    def __init__(
        self,
        input_size: int = (1, 138, 256),
        hidden_channels: list[int] = [8, 16, 32, 64, 128, 256],
        output_size: int = 9,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = True,
        kernel_size: int = 5,
        flatten_features: bool = False,
        pool_indices: list[int] = [1, 3, 5, 7],
        global_pool: str = "max",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.use_batchnorm = use_batchnorm
        self.kernel_size = kernel_size
        self.flatten_features = flatten_features
        self.pool_indices = pool_indices
        self.global_pool_type = global_pool

        # Create named blocks for feature extraction at different levels
        self.blocks = nn.ModuleDict()
        in_ch = input_size[0]
        for idx, out_ch in enumerate(hidden_channels):
            block = self.create_residual_block(in_ch, out_ch, idx)
            self.blocks[f"block_{idx}"] = block
            in_ch = out_ch

        # Classifier head
        if self.global_pool_type == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            # Use global max pooling instead of flatten for better features
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the pooled features
            nn.Linear(in_ch, in_ch // 2, bias=False),
            nn.BatchNorm1d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_ch // 2, output_size),
        )

        if output_size == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def create_residual_block(self, in_ch, out_ch, idx):
        """ResNet-inspired block with proper residual connections"""
        block = []

        # First conv
        block.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                self.kernel_size,
                padding=self.kernel_size // 2,
                bias=False,
            )
        )
        block.append(nn.BatchNorm2d(out_ch))
        # Attention after feature extraction
        block.append(SCSEModule(out_ch, reduction=max(4, out_ch // 4)))

        # Activation after attention
        block.append(nn.ReLU(inplace=True))

        # Conditional pooling
        if idx in self.pool_indices:
            block.append(nn.MaxPool2d(2, 2))

        # Dropout at the end
        if self.dropout_rate > 0:
            block.append(nn.Dropout2d(self.dropout_rate))

        return nn.Sequential(*block)

    @property
    def summary(self) -> str:
        """
        Generate a detailed summary of the CNN2D model.

        Returns:
            str: Formatted summary containing model architecture and parameters
        """
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("CNN2D Model Summary")
        summary_lines.append("=" * 80)

        # Model configuration
        summary_lines.append(f"Input Size: {self.input_size}")
        summary_lines.append(f"Hidden Channels: {self.hidden_channels}")
        summary_lines.append(f"Dropout Rate: {self.dropout_rate}")
        summary_lines.append(f"Kernel Size: {self.kernel_size}")
        summary_lines.append(f"Use BatchNorm: {self.use_batchnorm}")
        summary_lines.append(f"Total Blocks: {len(self.hidden_channels)}")

        # Block-wise details
        summary_lines.append("\nBlock Architecture:")
        summary_lines.append("-" * 50)
        total_params = 0

        for block_name, block in self.blocks.items():
            block_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
            total_params += block_params

            # Get input/output channels for this block
            idx = int(block_name.split("_")[1])
            in_ch = 1 if idx == 0 else self.hidden_channels[idx - 1]
            out_ch = self.hidden_channels[idx]

            summary_lines.append(
                f"{block_name:>10}: {in_ch:>3} -> {out_ch:>3} channels | "
                f"{block_params:>8,} params"
            )

        # Classifier details
        classifier_params = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )
        total_params += classifier_params

        classifier_in = self.hidden_channels[-1]  # After global pooling
        classifier_out = self.classifier[-1].out_features
        summary_lines.append(
            f"{'classifier':>10}: {classifier_in:>3} -> {classifier_out:>3}"
            f" units | {classifier_params:>8,} params"
        )

        # Parameter summary
        summary_lines.append("\nParameter Summary:")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Total Trainable Parameters: {total_params:,}")

        # Memory estimation (rough)
        param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        summary_lines.append(f"Estimated Model Size: {param_size_mb:.2f} MB")

        # Feature extraction info
        summary_lines.append("\nFeature Extraction:")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Available Blocks: {list(self.blocks.keys())}")
        summary_lines.append(f"Feature Pyramid Levels: {len(self.blocks)}")
        summary_lines.append(
            "Attention Module: SCSE (Spatial & Channel Squeeze-Excite)"
        )

        # Example input/output shapes for 2D
        summary_lines.append("\nExample Shapes (batch_size=1):")
        summary_lines.append("-" * 50)
        if isinstance(self.input_size, tuple) and len(self.input_size) == 3:
            c, h, w = self.input_size
            summary_lines.append(f"Input:  [1, {c}, {h}, {w}]")

            for idx, ch in enumerate(self.hidden_channels):
                summary_lines.append(f"Block {idx}: [1, {ch}, {h}, {w}]")
        else:
            summary_lines.append(f"Input:  [1, 1, H, W] (size: {self.input_size})")

            for idx, ch in enumerate(self.hidden_channels):
                summary_lines.append(f"Block {idx}: [1, {ch}, H, W]")

        # Show global max pooling effect
        last_ch = self.hidden_channels[-1]
        summary_lines.append(f"After global max pool: [1, {last_ch}, 1]")
        summary_lines.append(f"After flatten: [1, {last_ch}]")
        out_features = self.classifier[-1].out_features
        summary_lines.append(f"Output: [1, {out_features}]")

        summary_lines.append("=" * 80)

        return "\n".join(summary_lines)

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract features from all named blocks for feature pyramid training"""
        x = x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, L]

        features = {}
        for idx in range(len(self.hidden_channels)):
            block_name = f"block_{idx}"
            x = self.blocks[block_name](x)
            features[block_name] = x

        return features, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, features]
        x = x.unsqueeze(1) if x.dim() == 2 else x

        features, x = self.extract_features(x)

        # Apply global max pooling instead of flatten
        x = self.global_pool(x)
        logits = self.classifier(x.view(x.size(0), -1))  # [B, output_size]

        return logits, self.final_activation(logits)

    def get_block_names(self) -> list[str]:
        """Return list of block names for student-teacher training"""
        return [f"block_{idx}" for idx in range(len(self.hidden_channels))]

    def get_block(self, block_name: str) -> nn.Module:
        """Get specific block by name for feature matching"""
        return self.blocks[block_name]


if __name__ == "__main__":
    model = BlockCNN2D(
        input_size=(1, 138, 256),
        # hidden_channels=[32, 32, 64, 64, 128, 128, 256, 256, 128, 64, 32],
        output_size=9,
        dropout_rate=0.1,
    )
    inp = torch.randn(5, 1, 138, 256)
    out, probs = model(inp)
    # Print detailed model summary
    print(model.summary)

    model = CNN1D(
        input_size=46,
        hidden_channels=[32, 64, 128, 64, 32],
        output_size=9,
        dropout_rate=0.1,
    )

    # Print detailed model summary
    print(model.summary)
